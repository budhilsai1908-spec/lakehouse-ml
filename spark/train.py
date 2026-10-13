"""
train.py - Phase 3 of the lakehouse-ml project.

Reads the NYC taxi Iceberg table, builds features, trains a model to predict
trip duration (minutes), and logs params + metrics + the model to MLflow.
Trains two variants so they can be compared in the MLflow UI.
"""

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CATALOG = "lake"
TABLE = f"{CATALOG}.nyc.yellow_trips"
MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT = "taxi-trip-duration"
SAMPLE_ROWS = 200_000

FEATURES = ["trip_distance", "passenger_count", "PULocationID",
            "DOLocationID", "pickup_hour", "pickup_dow"]
TARGET = "trip_minutes"


def build_spark():
    b = SparkSession.builder.appName("train")
    b = b.config("spark.sql.extensions",
                 "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    b = b.config(f"spark.sql.catalog.{CATALOG}", "org.apache.iceberg.spark.SparkCatalog")
    b = b.config(f"spark.sql.catalog.{CATALOG}.type", "jdbc")
    b = b.config(f"spark.sql.catalog.{CATALOG}.uri",
                 "jdbc:postgresql://postgres:5432/catalog")
    b = b.config(f"spark.sql.catalog.{CATALOG}.jdbc.user", "iceberg")
    b = b.config(f"spark.sql.catalog.{CATALOG}.jdbc.password", "iceberg123")
    b = b.config(f"spark.sql.catalog.{CATALOG}.warehouse", "s3a://warehouse/")
    b = b.config(f"spark.sql.catalog.{CATALOG}.io-impl",
                 "org.apache.iceberg.aws.s3.S3FileIO")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.endpoint", "http://minio:9000")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.path-style-access", "true")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.access-key-id", "minioadmin")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.secret-access-key", "minioadmin123")
    return b.getOrCreate()


def load_dataframe(spark):
    df = (
        spark.table(TABLE)
        .withColumn("trip_minutes",
                    (unix_timestamp("tpep_dropoff_datetime")
                     - unix_timestamp("tpep_pickup_datetime")) / 60.0)
        .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", dayofweek("tpep_pickup_datetime"))
        .filter((col("trip_distance") > 0) & (col("trip_distance") < 100))
        .filter((col("trip_minutes") >= 1) & (col("trip_minutes") <= 120))
        .select(*FEATURES, TARGET)
        .dropna()
    )
    total = df.count()
    frac = min(1.0, SAMPLE_ROWS / total)
    return df.sample(fraction=frac, seed=42).toPandas()


def train_and_log(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_metric("mae_minutes", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"{name:18s}  MAE={mae:.2f} min   R2={r2:.3f}")
        return mae


def main():
    spark = build_spark()
    pdf = load_dataframe(spark)
    print(f"Training on {len(pdf):,} sampled rows.")

    X = pdf[FEATURES]
    y = pdf[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    train_and_log("linear_regression", LinearRegression(),
                  X_train, X_test, y_train, y_test)
    train_and_log("random_forest",
                  RandomForestRegressor(n_estimators=50, max_depth=12,
                                        n_jobs=-1, random_state=42),
                  X_train, X_test, y_train, y_test)

    spark.stop()
    print("Done. View runs at http://localhost:5000")


if __name__ == "__main__":
    main()
