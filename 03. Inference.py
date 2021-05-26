# Databricks notebook source
# MAGIC %md # 2. Distributed inference of Deep Learning models in Databricks

# COMMAND ----------

import mlflow
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import col, struct

# COMMAND ----------

# MAGIC %md ### A. Distributed Batch Inference

# COMMAND ----------

# DBTITLE 1,Load images from a test dataframe
# Test data. Using the same dataframe in this example
images_df = spark.read.table('dais_2021.flowers_train')
display(images_df)

# COMMAND ----------

# DBTITLE 1,Score incoming images
# Always use the Production version of the model from the registry
mlflow_model_path = 'models:/DAIS - Image classification/Production'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, mlflow_model_path, result_type=ArrayType(StringType()))

# Predict on a Spark DataFrame.
scored_df = (images_df
             .withColumn('origin', col("image.origin"))
             .withColumn('my_predictions', loaded_model(struct("origin")))
             .drop("origin"))
display(scored_df)

# COMMAND ----------

# DBTITLE 1,Save to scored location
# Write to the deltalake

# Image data is already compressed. So we turn off Parquet compression.
compression = spark.conf.get("spark.sql.parquet.compression.codec")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

(scored_df.write
 .format('delta')
 .mode('overwrite')
 .option('mergeSchema', True)
 .saveAsTable("dais_2021.flowers_predicted")
)

# Set compression back to snappy / earlier value
spark.conf.set("spark.sql.parquet.compression.codec", compression)

# COMMAND ----------

# MAGIC %md ### B. Distributed Streaming Inference

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table datacollablab.flowers_predicted_stream

# COMMAND ----------

# MAGIC %fs rm -r /dais/image_processing/inference_stream_chkpnt

# COMMAND ----------

# DBTITLE 1,Stream images from a Delta table to run inference on
# Using the same dataframe in this example
images_stream_source = (spark.readStream
                        .format("delta")
                        .option("maxFilesPerTrigger", 1)
                        .table('dais_2021.flowers_train')
                       )

# COMMAND ----------

# DBTITLE 1,Score incoming images in the stream
# Always use the Production version of the model from the registry
mlflow_model_path = 'models:/DAIS - Image classification model/Production'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, mlflow_model_path, result_type=ArrayType(StringType()))

scored_stream = (images_stream_source
                 .withColumn('origin', col("image.origin"))
                 .withColumn('my_predictions', loaded_model(struct("origin")))
                 .drop("origin"))

# COMMAND ----------

# DBTITLE 1,Save to scored location
# Write to the deltalake

# Image data is already compressed. So we turn off Parquet compression.
compression = spark.conf.get("spark.sql.parquet.compression.codec")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

(scored_stream.writeStream
 .format('delta')
 .outputMode("append")
 .option('mergeSchema', True)
 .option("checkpointLocation", "/dais/image_processing/inference_stream_chkpnt")
 .table("dais_2021.flowers_predicted_stream")
)

# Set compression back to snappy / earlier value
spark.conf.set("spark.sql.parquet.compression.codec", compression)

# COMMAND ----------

display(spark.table("dais_2021.flowers_predicted_stream"))
