from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityValidation").getOrCreate()

# Load model and validation data
model = LogisticRegressionModel.load("model/wine_quality_model")
val_data = spark.read.csv("data/ValidationDataset.csv", header=True, inferSchema=True)

# Preprocess validation data
feature_cols = [col for col in val_data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
val_data = assembler.transform(val_data)

# Validate the model
predictions = model.transform(val_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

print(f"Validation F1 Score: {f1_score}")
spark.stop()
