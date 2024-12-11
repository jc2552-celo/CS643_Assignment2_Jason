from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

# Load training data
data = spark.read.csv("data/TrainingDataset.csv", header=True, inferSchema=True)

# Preprocess data
feature_cols = [col for col in data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# Split dataset for internal validation
train_data, val_data = data.randomSplit([0.8, 0.2], seed=1234)

# Train logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='quality', maxIter=10)
model = lr.fit(train_data)

# Save the model
model.write().overwrite().save("model/wine_quality_model")

print("Model training completed and saved.")
spark.stop()
