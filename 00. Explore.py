# Databricks notebook source
# MAGIC %md
# MAGIC ![Overview](files/images/ImagePipelinePic.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Explore the image dataset

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
# MAGIC #### View Flower images data in a dataframe

# COMMAND ----------

# MAGIC %fs ls /datacollablab/flower_photos/

# COMMAND ----------

# MAGIC %fs ls /datacollablab/flower_photos/label=daisy/

# COMMAND ----------

display(spark.read.format("image").load("/datacollablab/flower_photos/"))

# COMMAND ----------

display(spark.read.format("image").load("dbfs:/datacollablab/flower_photos/label=sunflowers"))

# COMMAND ----------

# MAGIC %md #### Store the image data as a Delta table

# COMMAND ----------

path_labeled_train = "/datacollablab/flower_photos"

# COMMAND ----------

image_df = spark.read.format("image").load(path_labeled_train)
display(image_df)

# COMMAND ----------

image_df.write.format("delta").mode("overwrite").saveAsTable("dais_2021.flowers_train")

# COMMAND ----------

# MAGIC %md
# MAGIC #### We can view the image Delta table in data tab

# COMMAND ----------

display(spark.table("dais_2021.flowers_train"))

# COMMAND ----------

print("Number of images in training Delta table: " + str(spark.table("dais_2021.flowers_train").count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### For larger image datasets, load images into a DataFrame using binary file data source.
# MAGIC ##### Reference: https://docs.databricks.com/_static/notebooks/deep-learning/dist-img-infer-1-etl.html

# COMMAND ----------

# MAGIC %sql
# MAGIC optimize dais_2021.flowers_train
