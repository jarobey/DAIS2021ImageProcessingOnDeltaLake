# Databricks notebook source
# MAGIC %md # Simplify data conversion from Spark to PyTorch
# MAGIC 
# MAGIC This notebook demonstrates the following workflow on Databricks:
# MAGIC 1. Load data using Spark.
# MAGIC 2. Convert the Spark DataFrame to a PyTorch DataLoader using petastorm `spark_dataset_converter`.
# MAGIC 3. Feed the data into a single-node PyTorch model for training.
# MAGIC 4. Feed the data into a distributed PyTorch model for training.
# MAGIC 
# MAGIC The example in this notebook is based on the [transfer learning tutorial from PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). It applies the pre-trained [MobileNetV2](https://pytorch.org/docs/stable/torchvision/models.html#mobilenet-v2) model to the flowers dataset.
# MAGIC 
# MAGIC ### Requirements
# MAGIC 1. Databricks Runtime 7.0 ML +. On Databricks Runtime 6.x ML, you need to install petastorm==0.9.0 and pyarrow==0.15.0 on the cluster.
# MAGIC 2. Node type: one driver and two workers. Databricks recommends using GPU instances.
# MAGIC 
# MAGIC ##### Reference notebook: https://docs.databricks.com/applications/machine-learning/model-inference/resnet-model-inference-pytorch.html

# COMMAND ----------

# MAGIC %pip install pytorch-lightning==1.2.4

# COMMAND ----------

from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import torch
import torchvision
from PIL import Image
from functools import partial 
from petastorm import TransformSpec
from torchvision import transforms

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.torch as hvd
from sparkdl import HorovodRunner

import mlflow
import pytorch_lightning
from torch.utils.data import Dataset

# COMMAND ----------

BATCH_SIZE = 32
NUM_EPOCHS = 5

# COMMAND ----------

DIR = "/dais"
model_dir_path = "/DAIS - Image processing on Delta Lake & Databricks/ImageClassificationModel"

# COMMAND ----------

# Enable MLflow Tracking
mlflow.set_experiment(model_dir_path)
# MLflow will automatically track the runs using TensorFlow model
mlflow.pytorch.autolog()
mlflow.spark.autolog()

# Model name in MLflow Model Resigtry
model_name = "DAIS - Image classification"

# COMMAND ----------

# MAGIC %md ## 1. Load data using Spark
# MAGIC 
# MAGIC ### The flowers dataset
# MAGIC 
# MAGIC This example uses the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from the TensorFlow team,
# MAGIC which contains flower photos stored under five sub-directories, one per class. It is hosted under Databricks Datasets `dbfs:/databricks-datasets/flower_photos` for easy access.
# MAGIC 
# MAGIC The example loads the flowers table which contains the preprocessed flowers dataset using the binary file data source. It uses a small subset of the flowers dataset, including ~90 training images and ~10 validation images to reduce the running time of this notebook. When you run this notebook, you can increase the number of images used for better model accuracy.

# COMMAND ----------

df = spark.read.format("delta").load("/databricks-datasets/flowers/delta") \
  .select(col("content"), col("label_index")) \
  .limit(100)
  
num_classes = df.select("label_index").distinct().count()
df_train, df_val = df.randomSplit([0.9, 0.1], seed=12345)

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cache the Spark DataFrame using Petastorm Spark converter

# COMMAND ----------

# Set a cache directory on DBFS FUSE for intermediate data.
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/" + DIR + "petastorm_cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feed the data into a single-node PyTorch model for training
# MAGIC 
# MAGIC ### Get the model MobileNetV2 from torchvision

# COMMAND ----------

# First, load the model and inspect the structure of the model.
torchvision.models.mobilenet_v2(pretrained=True)

# COMMAND ----------

def get_model(lr=0.001):
  # Load a MobileNetV2 model from torchvision
  model = torchvision.models.mobilenet_v2(pretrained=True)
  # Freeze parameters in the feature extraction layers
  for param in model.parameters():
    param.requires_grad = False
    
  # Add a new classifier layer for transfer learning
  num_ftrs = model.classifier[1].in_features
  # Parameters of newly constructed modules have requires_grad=True by default
  model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
  
  return model

# COMMAND ----------

# MAGIC %md ### Define the train and evaluate function for the model

# COMMAND ----------

def train_one_epoch(model, criterion, optimizer, scheduler, 
                    train_dataloader_iter, steps_per_epoch, epoch, 
                    device):
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for step in range(steps_per_epoch):
    pd_batch = next(train_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label_index'].to(device)
    
    # Track history in training
    with torch.set_grad_enabled(True):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      # backward + optimize
      loss.backward()
      optimizer.step()

    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
  
  scheduler.step()

  epoch_loss = running_loss / (steps_per_epoch * BATCH_SIZE)
  epoch_acc = running_corrects.double() / (steps_per_epoch * BATCH_SIZE)

  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

def evaluate(model, criterion, val_dataloader_iter, validation_steps, device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over all the validation data.
  for step in range(validation_steps):
    pd_batch = next(val_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label_index'].to(device)

    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  
  # Average the losses across observations for each minibatch.
  epoch_loss = running_loss / validation_steps
  epoch_acc = running_corrects.double() / (validation_steps * BATCH_SIZE)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

# MAGIC %md ### Preprocess images
# MAGIC 
# MAGIC Before feeding the dataset into the model, you need to decode the raw image bytes and apply standard ImageNet transforms. Databricks recommends not doing this transformation on the Spark DataFrame since that substantially increases the size of the intermediate files and might decrease performance. Instead, do this transformation in a `TransformSpec` function in petastorm.

# COMMAND ----------

def transform_row(is_train, pd_batch):
  """
  The input and output of this function must be pandas dataframes.
  Do data augmentation for the training dataset only.
  """
  transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
  if is_train:
    transformers.extend([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
    ])
  else:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
    ])
  transformers.extend([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  
  trans = transforms.Compose(transformers)
  
  pd_batch['features'] = pd_batch['content'].map(lambda x: trans(x).numpy())
  pd_batch = pd_batch.drop(labels=['content'], axis=1)
  return pd_batch

def get_transform_spec(is_train=True):
  # The output shape of the `TransformSpec` is not automatically known by petastorm, 
  # so you need to specify the shape for new columns in `edit_fields` and specify the order of 
  # the output columns in `selected_fields`.
  return TransformSpec(partial(transform_row, is_train), 
                       edit_fields=[('features', np.float32, (3, 224, 224), False)], 
                       selected_fields=['features', 'label_index'])

# COMMAND ----------

# MAGIC %md ### Train and evaluate the model on the local machine
# MAGIC 
# MAGIC Use `converter.make_torch_dataloader(...)` to create the dataloader.

# COMMAND ----------

# Create a custom PyTorch dataset class
class ImageDataset(Dataset):
  """Create a custom PyTorch dataset class."""
  def __init__(self, paths, transform=None):
    self.paths = paths
    self.transform = transform
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    from torchvision.datasets.folder import default_loader
    image = default_loader(self.paths[index])
    if self.transform is not None:
      image = self.transform(image)
    return image

# COMMAND ----------

# TIP: Create custom Python pyfunc model that transforms and predicts on inference data
# Allows the inference pipeline to be independent of the model framework used in training pipeline
class pytorchCNNModelWrapper(mlflow.pyfunc.PythonModel):
  """Create custom Python pyfunc model that transforms and predicts on inference data.
  Allows the inference pipeline to be independent of the model framework used in training pipeline.
  """
  def __init__(self, model_path):
    self.model_path = model_path

  def load_context(self, context):
    # Load the PyTorch-native representation of the MLflow model
    self.model = mlflow.pytorch.load_model(
        model_uri=self.model_path)

  def predict(self, context, model_input):
    #import torch
    import json
    import pandas as pd

    class_def = {
      0: 'sunflowers', 
      1: 'tulips', 
      2: 'daisy', 
      3: 'dandelion', 
      4: 'roses'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_input['origin'] = model_input['origin'].str.replace("dbfs:","/dbfs")
    paths = model_input['origin']

    rtn_df = model_input.iloc[:,0:1]
    rtn_df['prediction'] = None
    rtn_df['probabilities'] = None

    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    #,
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                   std=[0.229, 0.224, 0.225])
    ])
    images = ImageDataset(paths, transform=transform)
    
    loader = torch.utils.data.DataLoader(images, batch_size=500, num_workers=8)
    model = self.model
    model.to(device)
    all_predictions = []
    with torch.no_grad():
      for batch in loader:
        predictions = list(model(batch.to(device)).cpu().numpy())
        for prediction in predictions:
          all_predictions.append(prediction)
    
    rtn_df['prediction'] = [np.argmax(i, axis=0) for i in all_predictions]
    pred_probs_num = [dict(enumerate(map(str, i))) for i in all_predictions]
    rtn_df['probabilities'] = [json.dumps({class_def[k]: v for k, v in row.items()}) for row in pred_probs_num]
    
    return rtn_df[['prediction', 'probabilities']].values.tolist()

# COMMAND ----------

def train_and_evaluate(lr=0.001):
  from mlflow.tracking import MlflowClient
  
  with mlflow.start_run() as run:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(lr=lr)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Only parameters of final layer are being optimized.
    optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                               batch_size=BATCH_SIZE) as train_dataloader, \
         converter_val.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), 
                                             batch_size=BATCH_SIZE) as val_dataloader:

      train_dataloader_iter = iter(train_dataloader)
      steps_per_epoch = len(converter_train) // BATCH_SIZE

      val_dataloader_iter = iter(val_dataloader)
      validation_steps = max(1, len(converter_val) // BATCH_SIZE)

      for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, exp_lr_scheduler, 
                                                train_dataloader_iter, steps_per_epoch, epoch, 
                                                device)
        val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps, device)
        
      # Log the native Pytorch model in MLFlow
      run_id = mlflow.active_run().info.run_id
      experiment_id = mlflow.active_run().info.experiment_id
      mlflow.pytorch.log_model(model, 'pytorch_model_v1')
      
      # Log the wrapped model as a custom pyfunc model in MLFlow
      wrappedModel = pytorchCNNModelWrapper(f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/pytorch_model_v1")
      mlflow.pyfunc.log_model("pyfunc_pytorch_model_v2", python_model=wrappedModel, registered_model_name=model_name)
      #print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")
      
      # Transition the generic pyfunc model from 'None' version to 'Staging' version
      client = MlflowClient()
      max_model_version = max([int(each.version) for each in client.search_model_versions(f"name='{model_name}'")])
      client.transition_model_version_stage(
          name=model_name,
          version=max_model_version,
          stage="Staging"
      )
      
      mlflow.end_run()

  return val_loss
  
loss = train_and_evaluate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feed the data into a distributed PyTorch model for training.
# MAGIC 
# MAGIC Use HorovodRunner for distributed training.
# MAGIC 
# MAGIC The example uses the default value of parameter `num_epochs=None` to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where you need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee.

# COMMAND ----------

def metric_average(val, name):
  tensor = torch.tensor(val)
  avg_tensor = hvd.allreduce(tensor, name=name)
  return avg_tensor.item()

def train_and_evaluate_hvd(lr=0.001):
  hvd.init()  # Initialize Horovod.
  
  # Horovod: pin GPU to local rank.
  if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    device = torch.cuda.current_device()
  else:
    device = torch.device("cpu")
  
  model = get_model(lr=lr)
  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()
  
  # Effective batch size in synchronous distributed training is scaled by the number of workers.
  # An increase in learning rate compensates for the increased batch size.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr * hvd.size(), momentum=0.9)
  
  # Broadcast initial parameters so all workers start with the same parameters.
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  
  # Wrap the optimizer with Horovod's DistributedOptimizer.
  optimizer_hvd = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hvd, step_size=7, gamma=0.1)

  with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             cur_shard=hvd.rank(), shard_count=hvd.size(),
                                             batch_size=BATCH_SIZE) as train_dataloader, \
       converter_val.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False),
                                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                                           batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = max(1, len(converter_val) // (BATCH_SIZE * hvd.size()))
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train_one_epoch(model, criterion, optimizer_hvd, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps,
                                   device, metric_agg_fn=metric_average)

  return val_loss

# COMMAND ----------

with mlflow.start_run():
  hr = HorovodRunner(np=2)   # This assumes the cluster consists of two workers.
  hr.run(train_and_evaluate_hvd)
