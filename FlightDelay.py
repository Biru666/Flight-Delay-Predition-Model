import sys
import time
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, StandardScaler, 
    VectorAssembler, PCA
)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def build_pipeline(df):
    # Handle categorical features
    cat_cols = ["Operated_or_Branded_Code_Share_Partners", "DepTimeBlk", "ArrTimeBlk"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_Idx") for c in cat_cols]
    
    # Build Pipelines
    feature_cols = [c + "_Idx" for c in cat_cols] + [col for col in df.columns if col not in ["ArrDel15"]+ cat_cols]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
     
    # Apply PCA
    #scaler = StandardScaler(inputCol="features", outputCol="scaledfeatures")
    #pca = PCA(k=10, inputCol="scaledfeatures", outputCol="pcafeatures")    
    #dt = DecisionTreeClassifier(featuresCol="pcafeatures", labelCol="ArrDel15")
    #pipeline = Pipeline(stages= indexers + [assembler, scaler, pca, dt])

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="ArrDel15")

    pipeline = Pipeline(stages= indexers + [assembler, dt])

    
    return pipeline

def main():    
    spark = SparkSession.builder.appName("FlightDelay Prediction").getOrCreate()
    sc = spark.sparkContext

    start_time = time.time() 

    # Create DataFrame
    df = spark.read.csv("hdfs:///user/linbiru/input/Combined_Flights_2022.csv", header=True, inferSchema=True)


    # data preprocess
    df = df.na.drop()
    df_clean = df.drop("FlightDate", "Year", "Diverted", "Cancelled", "DestState", "Dest", 
    "DestStateName", "Operating_Airline", "Tail_Number", "IATA_Code_Marketing_Airline", 
    "Marketing_Airline_Network", "Origin", "Airline", "IATA_Code_Operating_Airline",
    "OriginCityName", "OriginState", "OriginStateName", "DestCityName", "AirTime", 
    "ArrTime", "ArrDelay", "ArrDelayMinutes", "ActualElapsedTime", 
    "TaxiOut", "ArrivalDelayGroups")    

    pipeline = build_pipeline(df_clean)
    
    # split data
    train_data, test_data = df_clean.randomSplit([0.8, 0.2])

    # Train model
    model = pipeline.fit(train_data) 

    dt_model = model.stages[-1]
    #feature_cols = model.stages[-2].getInputCol()
    feature_cols = model.stages[-2].getInputCols()
    feature_importances = dt_model.featureImportances.toArray()
    
    # Evaluate
    train_predictions = model.transform(train_data)
    test_predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="ArrDel15", 
                                            predictionCol="prediction", metricName="accuracy")
    train_accuracy = evaluator.evaluate(train_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    #outputs
    output = sc.parallelize(
    [
        f"Training Accuracy: {train_accuracy:.4f}",
        f"Test Accuracy: {test_accuracy:.4f}",
        f"Running Time: {time.time() - start_time:.4f} seconds"
    ]
    + [f"{name}: {score:.4f}" for name, score in zip(feature_cols, feature_importances)]
    )
    output.saveAsTextFile(f"hdfs:///user/linbiru/output/flightdelays")    
    spark.stop()

if __name__ == "__main__":
    main()
