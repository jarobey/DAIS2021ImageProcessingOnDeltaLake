# Databricks notebook source
# MAGIC %md
# MAGIC # 1. ETL images into a Delta table

# COMMAND ----------

import io
import numpy as np
import pandas as pd
import uuid
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from PIL import Image

# COMMAND ----------

# MAGIC %md ### Sample data: The flowers dataset
# MAGIC 
# MAGIC We use the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from the TensorFlow team as our example dataset,
# MAGIC which contains flower photos stored under five sub-directories, one per class.
# MAGIC It is hosted under Databricks Datasets for easy access.
# MAGIC 
# MAGIC * Data owner: https://www.tensorflow.org/datasets/catalog/tf_flowers
# MAGIC * License: https://creativecommons.org/licenses/by/4.0/

# COMMAND ----------

# MAGIC %md
# MAGIC #### For larger image datasets, load images into a DataFrame using binary file data source.
# MAGIC ##### Reference: https://docs.databricks.com/_static/notebooks/deep-learning/dist-img-infer-1-etl.html

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.jpg") \
  .load("/databricks-datasets/flower_photos")
display(images)

# COMMAND ----------

# MAGIC %md ###Expand the DataFrame with extra metadata columns.
# MAGIC 
# MAGIC We extract some frequently used metadata from `images` DataFrame:
# MAGIC * extract labels from file paths,
# MAGIC * add label indices,
# MAGIC * extract image sizes.

# COMMAND ----------

def extract_label(path_col):
  """Extract label from file path using built-in SQL functions."""
  return regexp_extract(path_col, "flower_photos/([^/]+)", 1)

def extract_size(content):
  """Extract image size from its raw content."""
  image = Image.open(io.BytesIO(content))
  return image.size

@pandas_udf("width: int, height: int")
def extract_size_udf(content_series):
  sizes = content_series.apply(extract_size)
  return pd.DataFrame(list(sizes))

# COMMAND ----------

images_with_label = images.select(
  col("path"),
  extract_label(col("path")).alias("label"),
  extract_size_udf(col("content")).alias("size"),
  col("content"))
display(images_with_label)

# COMMAND ----------

# DBTITLE 1,Transform label to index
labels = images_with_label.select(col("label")).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}
num_classes = len(label_to_idx)

@pandas_udf("long")
def get_label_idx(labels):
  return labels.map(lambda label: label_to_idx[label])

# COMMAND ----------

df = (images_with_label
      .withColumn("label_index", get_label_idx(col("label")))
      .select(col("path"), col("size"), col("label"), col("label_index"), col("content")))

# COMMAND ----------

print("Number of images in binary Delta table: " + str(df.count()))

# COMMAND ----------

# DBTITLE 1,You can also do other processing using python native functions by distributing it and running it in parallel over groups of the dataset
# Reference: https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html

from pyspark.sql.functions import lit
from pyspark.sql.types import BinaryType,StringType

def get_image_bytes(image):
  img_bytes = io.BytesIO()
  image.save(img_bytes,format="png")
  return img_bytes.getvalue()

# Sample python native function that can do additional processing - expects pandas df as input and returns pandas df as output.
def add_grayscale_img(input_df):
  # Set up return frame.  In this case I'll have a row per passed in row.  You could be aggregating down to a single image, slicing
  # out columns,or just about anything, here.  For this case, I am simply going to return the input_df with some extra columns.
  input_df['grayscale_image'] = input_df.content.apply(lambda image: get_image_bytes(Image.open(io.BytesIO(image)).convert('L'))) 
  input_df['grayscale_format'] = "png" # Since this is a pandas df, this will assigne png to all rows
  
  return input_df
                                                

# Setup the return schema. Add blank columns to match the schema expected after applying the transformation function. Makes the schema definition easy in the function invocation.
rtn_schema = (df.select('content','label','path')
                                 .withColumn('grayscale_image', lit(None).cast(BinaryType()))
                                 .withColumn('grayscale_format', lit(None).cast(StringType()))
                                )
# Reduce df down to data used in the function, the groupBy, and the re-join key respectively.  This could include other features as used by your pandas function
limited_df = df.select('content','label','path')
                     
# Returns spark dataframe with transformations applied in parallel for each 'group'
augmented_df = limited_df.groupBy('label').applyInPandas(add_grayscale_img, schema=rtn_schema.schema)
                     
# re-join to the full dataset using leftouter in case the image transform needed to skip some rows
output_df = df.join(augmented_df.select('label','grayscale_image','grayscale_format'),['label'],"leftouter")                     

# COMMAND ----------

# DBTITLE 1,Setup some test data for function debug
# Setup some test data so I can iterate quickly on my image processing function
pd_df = limited_df.limit(10).toPandas()
print(pd_df.columns)

# COMMAND ----------

# DBTITLE 1,Make sure function works correctly
# Some testing code
test_df = pd_df.copy()
add_grayscale_img(test_df)
print(test_df['grayscale_image'])

# display one image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

color_image = mpimg.imread(io.BytesIO(test_df.loc[1,'content']), format='jpg')
image = mpimg.imread(io.BytesIO(test_df.loc[1,'grayscale_image']), format='png')
print('color dimensions = {}'.format(color_image.shape))
print('grayscale dimensions = {}'.format(image.shape))

row_count = test_df.count()[0]
plt.figure(figsize=(8,20))
for idx,row in test_df.iterrows():
  (content,_,_,grayscale,_) = row
  color_image = mpimg.imread(io.BytesIO(content), format='jpg')
  image = mpimg.imread(io.BytesIO(grayscale), format='png')
  plt.subplot(row_count,2,idx*2+1)
  plt.imshow(color_image)
  plt.subplot(row_count,2,idx*2+2)
  plt.imshow(image,cmap='gray')

# COMMAND ----------

# DBTITLE 1,See what we got
display(output_df)

# COMMAND ----------

# DBTITLE 1,Save the DataFrame in Delta format
# Image data is already compressed. So we turn off Parquet compression.
compression = spark.conf.get("spark.sql.parquet.compression.codec")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

# Save image data stored in binary format as a delta table
output_df.write.format("delta").mode("overwrite").option("mergeSchema",True).saveAsTable("dais_2021.flowers_train_binary")

# Set compression back to snappy / earlier value
spark.conf.set("spark.sql.parquet.compression.codec", compression)

# COMMAND ----------

# DBTITLE 1,Avoid the small files problem and improve the performance of large Delta tables using Optimize command
# MAGIC %sql
# MAGIC OPTIMIZE dais_2021.flowers_train_binary
