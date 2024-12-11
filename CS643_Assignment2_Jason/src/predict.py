import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler

if len(sys.argv) != 2:
    print("Usage: python predict.py <test_file_path>")
    sys.exit(1)

# Load test file path
test_file_path = sys.argv[1]

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load model and test data
model = LogisticRegressionModel.load("model/wine_quality_model")
test_data = spark.read.csv(test_file_path, header=True, inferSchema=True)

# Preprocess test data
feature_cols = [col for col in test_data.columns if col != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
test_data = assembler.transform(test_data)

# Perform predictions
predictions = model.transform(test_data)
predictions.select("features", "quality", "prediction").show()

spark.stop()
