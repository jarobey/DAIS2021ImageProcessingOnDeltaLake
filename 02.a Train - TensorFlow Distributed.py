# Databricks notebook source
# MAGIC %md # 1. Distributed training of Deep Learning models in Databricks
# MAGIC 
# MAGIC This notebook demonstrates the following workflow on Databricks:
# MAGIC 1. Load data using Spark.
# MAGIC 2. Convert the Spark DataFrame to a TensorFlow Dataset using petastorm `spark_dataset_converter`.
# MAGIC 3. Feed the data into a single-node TensorFlow model for training.
# MAGIC 4. Feed the data into a distributed TensorFlow model for training.
# MAGIC 
# MAGIC The example in this notebook is based on the [transfer learning tutorial from TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning). It applies the pre-trained [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) model to the [TensorFlow Flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers).
# MAGIC License: https://creativecommons.org/licenses/by/4.0/
# MAGIC 
# MAGIC ### Requirements
# MAGIC 1. Databricks Runtime 7.0 ML +. On Databricks Runtime 6.x ML, you need to install petastorm==0.9.0 and pyarrow==0.15.0 on the cluster.
# MAGIC 2. Node type: one driver and two workers. Databricks recommends using GPU instances.
# MAGIC 
# MAGIC ##### Reference notebook: https://docs.databricks.com/applications/machine-learning/load-data/petastorm.html#petastorm-tensorflow

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from petastorm import TransformSpec
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow as tf
# import tensorflow_hub as hub

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

import mlflow

# COMMAND ----------

DIR = "/dais"
model_dir_path = "/DAIS - Image processing on Delta Lake & Databricks/ImageClassificationModel"

# Enable MLflow Tracking
mlflow.set_experiment(model_dir_path)
# MLflow will automatically track the runs using TensorFlow model
mlflow.tensorflow.autolog()
mlflow.spark.autolog()

# COMMAND ----------

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 20
NUM_EPOCHS = 25

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

# Read the training data stored in Delta table
df = (spark.read.format("delta").table("dais_2021.flowers_train_binary")
      .select(col("content"), col("label_index").cast(LongType()))
      .limit(100))

num_classes = df.select("label_index").distinct().count()
df_train, df_val = df.randomSplit([0.6, 0.4], seed=12345)

# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cache the Spark DataFrame using Petastorm Spark converter

# COMMAND ----------

# Set a cache directory on DBFS FUSE for intermediate data.
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/" + DIR + "petastorm_cache")

# TIP: Use a low value for parquet_row_group_size_bytes. The default of 32 MiB can be too high for larger datasets. Using 1MB instead.
converter_train = make_spark_converter(df_train, parquet_row_group_size_bytes=1000000)
converter_val = make_spark_converter(df_val, parquet_row_group_size_bytes=1000000)

# COMMAND ----------

print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feed the data into a single-node TensorFlow model for training
# MAGIC 
# MAGIC ### Get the model MobileNetV2 from tensorflow.keras

# COMMAND ----------

# First, load the model and inspect the structure of the model.
MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet').summary()

# COMMAND ----------

def get_model(lr=0.001):

  # Create the base model from the pre-trained model MobileNet V2
  base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
  # Freeze parameters in the feature extraction layers
  base_model.trainable = False
  
  # Add a new classifier layer for transfer learning
  global_average_layer = keras.layers.GlobalAveragePooling2D()
  prediction_layer = keras.layers.Dense(num_classes)
  
  model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
  ])
  return model

def get_compiled_model(lr=0.001):
  model = get_model(lr=lr)
  #model.build(IMG_SHAPE)
  model.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

# COMMAND ----------

get_compiled_model().summary()

# COMMAND ----------

# MAGIC %md ### Preprocess images
# MAGIC 
# MAGIC Before feeding the dataset into the model, you need to decode the raw image bytes and apply standard ImageNet transforms. Databricks recommends not doing this transformation on the Spark DataFrame since that substantially increases the size of the intermediate files and might decrease performance. Instead, do this transformation in a `TransformSpec` function in petastorm.
# MAGIC 
# MAGIC Alternatively, you can also apply the transformation to the TensorFlow Dataset returned by the converter using `dataset.map()` with `tf.map_fn()`.

# COMMAND ----------

def preprocess(content):
  """
  Preprocess an image file bytes for MobileNetV2 (ImageNet).
  """
  image = Image.open(io.BytesIO(content)).resize([224, 224])
  image_array = keras.preprocessing.image.img_to_array(image)
  return preprocess_input(image_array)

def transform_row(pd_batch):
  """
  The input and output of this function are pandas dataframes.
  """
  pd_batch['features'] = pd_batch['content'].map(lambda x: preprocess(x))
  pd_batch = pd_batch.drop(labels='content', axis=1)
  return pd_batch

# The output shape of the `TransformSpec` is not automatically known by petastorm, 
# so you need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
transform_spec_fn = TransformSpec(
  transform_row, 
  edit_fields=[('features', np.float32, IMG_SHAPE, False)], 
  selected_fields=['features', 'label_index']
)

# COMMAND ----------

# MAGIC %md ### Train and evaluate the model on the local machine
# MAGIC 
# MAGIC Use `converter.make_tf_dataset(...)` to create the dataset.

# COMMAND ----------

# TIP: Create custom Python pyfunc model that transforms and predicts on inference data
# Allows the inference pipeline to be independent of the model framework used in training pipeline
class KerasCNNModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model_path):
    self.model_path = model_path

  def load_context(self, context):
    # Load the Keras-native representation of the MLflow
    # model
    print(self.model_path)
    self.model = mlflow.keras.load_model(
        model_uri=self.model_path)

  def predict(self, context, model_input):
    import tensorflow as tf
    import json

    class_def = {
      0: 'sunflowers', 
      1: 'tulips', 
      2: 'daisy', 
      3: 'dandelion', 
      4: 'roses'
    }

    model_input['origin'] = model_input['origin'].str.replace("dbfs:","/dbfs")
    images = model_input['origin']

    rtn_df = model_input.iloc[:,0:1]
    rtn_df['prediction'] = None
    rtn_df['probabilities'] = None

    for index, row in model_input.iterrows():
      image = np.round(np.array(Image.open(row['origin']).resize((224,224)),dtype=np.float32))
      img = tf.reshape(image, shape=[-1, 224, 224, 3])
      class_probs = self.model.predict(img)
      classes = np.argmax(class_probs, axis=1)
      class_prob_dict = dict()
      for key, val in class_def.items():
        class_prob_dict[val] = np.round(np.float(class_probs[0][int(key)]), 3).tolist()
      rtn_df.loc[index,'prediction'] = classes[0]
      rtn_df.loc[index,'probabilities'] = json.dumps(class_prob_dict)

    return rtn_df[['prediction', 'probabilities']].values.tolist()

# COMMAND ----------

def train_and_evaluate(lr=0.001):
  model = get_compiled_model(lr)
  
  with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       batch_size=BATCH_SIZE) as train_dataset, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     batch_size=BATCH_SIZE) as val_dataset:
    # tf.keras only accept tuples, not namedtuples
    train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
    steps_per_epoch = len(converter_train) // BATCH_SIZE
    print("steps_per_epoch: ")
    print(steps_per_epoch)

    val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
    validation_steps = max(1, len(converter_val) // BATCH_SIZE)
    print("validation_steps: ")
    print(validation_steps)
    
    print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

    hist = model.fit(train_dataset, 
                     steps_per_epoch=steps_per_epoch,
                     epochs=NUM_EPOCHS,
                     validation_data=val_dataset,
                     validation_steps=validation_steps,
                     verbose=2)
    
    # Note: Below is how you explicitly save a model.
    #mlflow.keras.log_model(model, 'keras_model_v1')

    # Log the custom pyfunc model
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id
    wrappedModel = KerasCNNModelWrapper(f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/model")
    mlflow.pyfunc.log_model("pyfunc_model_v2", python_model=wrappedModel)
    print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")
    mlflow.end_run()
    
  return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1] 
  
with mlflow.start_run() as r:
  loss, accuracy = train_and_evaluate()
  print("Validation Accuracy: {}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feed the data into a distributed TensorFlow model for training
# MAGIC 
# MAGIC Use HorovodRunner for distributed training.
# MAGIC 
# MAGIC Use the default value of parameter `num_epochs=None` to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where you need to guarantee that the numbers of data records seen on all workers are identical per step. Given that the length of each data shard may not be identical, setting `num_epochs` to any specific number would fail to meet the guarantee.

# COMMAND ----------

@tf.autograph.experimental.do_not_convert
def train_and_evaluate_hvd(lr=0.001):
  
  hvd.init()  # Initialize Horovod.
  
  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  model = get_model(lr)
  
  # Horovod: adjust learning rate based on number of GPUs.
  optimizer = keras.optimizers.SGD(lr=lr * hvd.size(), momentum=0.9)
  dist_optimizer = hvd.DistributedOptimizer(optimizer)
  
  callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
  ]
  
  # Set experimental_run_tf_function=False in TF 2.x
  model.compile(optimizer=dist_optimizer, 
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=["accuracy"],
                experimental_run_tf_function=False)
    
  with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       cur_shard=hvd.rank(), shard_count=hvd.size(),
                                       batch_size=BATCH_SIZE) as train_dataset, \
       converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                     cur_shard=hvd.rank(), shard_count=hvd.size(),
                                     batch_size=BATCH_SIZE) as val_dataset:
    # tf.keras only accept tuples, not namedtuples
    train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
    steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())

    val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
    validation_steps = max(1, len(converter_val) // (BATCH_SIZE * hvd.size()))
    
    hist = model.fit(train_dataset, 
                     steps_per_epoch=steps_per_epoch,
                     epochs=NUM_EPOCHS,
                     validation_data=val_dataset,
                     validation_steps=validation_steps,
                     callbacks=callbacks,
                     verbose=2)
    
  return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

with mlflow.start_run():
  hr = HorovodRunner(np=2)  # It assumes the cluster consists of two workers.
  hr.run(train_and_evaluate_hvd)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register the best model into MLflow registry

# COMMAND ----------

# MAGIC %md
# MAGIC #### Example workflow of image processing on delta table containing path to files instead of storing the full images themselves
# MAGIC  - Notebook: https://databricks.com/notebooks/wsi-image-segmentation-transfer-pandasudf.html
# MAGIC  - Blog: https://databricks.com/blog/2020/01/31/automating-digital-pathology-image-analysis-with-machine-learning-on-databricks.html
