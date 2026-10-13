"""
load_taxi.py - Phase 2 of the lakehouse-ml project.

Loads NYC Yellow Taxi trip data into a partitioned Apache Iceberg table.
Storage: MinIO (S3-compatible). Catalog: Postgres (JDBC). Engine: Spark.

Run inside the Spark container:

    docker exec -it lakehouse-spark /opt/spark/bin/spark-submit \
      --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.apache.hadoop:hadoop-aws:3.3.4,software.amazon.awssdk:bundle:2.24.6,org.postgresql:postgresql:42.7.3 \
      --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
      /opt/spark/work-dir/jobs/load_taxi.py

Note: AWS_REGION and the MinIO keys are provided to the container via the
compose file (the AWS SDK requires a region even though MinIO ignores it).
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# --- Config ---------------------------------------------------------------
CATALOG = "lake"
WAREHOUSE = "s3a://warehouse/"
MINIO_ENDPOINT = "http://minio:9000"
MINIO_KEY = "minioadmin"
MINIO_SECRET = "minioadmin123"
PG_URI = "jdbc:postgresql://postgres:5432/catalog"
PG_USER = "iceberg"
PG_PASSWORD = "iceberg123"

NAMESPACE = f"{CATALOG}.nyc"
TABLE = f"{NAMESPACE}.yellow_trips"
SOURCE = "/opt/spark/work-dir/data/yellow_tripdata_2024-01.parquet"


def build_spark():
    """Create a SparkSession wired to the Iceberg catalog on Postgres + MinIO."""
    b = SparkSession.builder.appName("load_taxi")
    b = b.config("spark.sql.extensions",
                 "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    b = b.config(f"spark.sql.catalog.{CATALOG}",
                 "org.apache.iceberg.spark.SparkCatalog")
    b = b.config(f"spark.sql.catalog.{CATALOG}.type", "jdbc")
    b = b.config(f"spark.sql.catalog.{CATALOG}.uri", PG_URI)
    b = b.config(f"spark.sql.catalog.{CATALOG}.jdbc.user", PG_USER)
    b = b.config(f"spark.sql.catalog.{CATALOG}.jdbc.password", PG_PASSWORD)
    b = b.config(f"spark.sql.catalog.{CATALOG}.warehouse", WAREHOUSE)
    b = b.config(f"spark.sql.catalog.{CATALOG}.io-impl",
                 "org.apache.iceberg.aws.s3.S3FileIO")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.endpoint", MINIO_ENDPOINT)
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.path-style-access", "true")
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.access-key-id", MINIO_KEY)
    b = b.config(f"spark.sql.catalog.{CATALOG}.s3.secret-access-key", MINIO_SECRET)
    return b.getOrCreate()


def create_table(spark):
    """Create the namespace and the partitioned Iceberg table if absent."""
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NAMESPACE}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            VendorID              INT,
            tpep_pickup_datetime  TIMESTAMP_NTZ,
            tpep_dropoff_datetime TIMESTAMP_NTZ,
            passenger_count       BIGINT,
            trip_distance         DOUBLE,
            PULocationID          INT,
            DOLocationID          INT,
            payment_type          BIGINT,
            fare_amount           DOUBLE,
            tip_amount            DOUBLE,
            total_amount          DOUBLE,
            pickup_date           DATE
        )
        USING iceberg
        PARTITIONED BY (pickup_date)
    """)


def load(spark):
    """Read the source parquet, trim + derive pickup_date, append to Iceberg."""
    df = spark.read.parquet(SOURCE)
    clean = (
        df.select(
            "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "passenger_count", "trip_distance", "PULocationID", "DOLocationID",
            "payment_type", "fare_amount", "tip_amount", "total_amount",
        )
        .withColumn("pickup_date", to_date(col("tpep_pickup_datetime")))
        .filter(col("pickup_date").isNotNull())
    )
    clean.writeTo(TABLE).append()
    return clean.count()


def main():
    spark = build_spark()
    create_table(spark)
    rows = load(spark)
    total = spark.sql(f"SELECT COUNT(*) FROM {TABLE}").collect()[0][0]
    print(f"Appended {rows:,} rows. Table now holds {total:,} rows.")
    spark.stop()


if __name__ == "__main__":
    main()
