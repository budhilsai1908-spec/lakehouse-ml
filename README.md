# lakehouse-ml

An end-to-end lakehouse + ML serving pipeline, built to run reproducibly on
Linux/Docker. Big-data storage with Apache Iceberg, orchestrated with Docker
Compose.

**Target stack:** Spark + Flink -> Iceberg (on MinIO) -> MLflow -> Ray Serve -> secured API

## Current status

- [x] Phase 1 - Infrastructure (MinIO + Postgres on Docker)
- [x] Phase 2 - Spark loads NYC taxi data into a partitioned Iceberg table
- [ ] Phase 3 - ML lifecycle with MLflow
- [ ] Phase 4 - Flink streaming ingest
- [ ] Phase 5 - Ray Serve model serving
- [ ] Phase 6 - Security layer (auth, TLS, least-privilege DB)

## Architecture

| Layer | Component | Role |
|---|---|---|
| Orchestration | Docker Compose (Linux) | Reproducible multi-service stack |
| Object storage | MinIO | S3-compatible store for Iceberg data + metadata |
| Table format | Apache Iceberg | Partitioning, time travel, schema evolution |
| Catalog | Postgres (JDBC catalog) | Tracks which tables exist and where |
| Batch engine | Spark (PySpark) | Loads and transforms the data |

## Setup

### 1. Bring up the stack

    docker compose --env-file .env -f compose/docker-compose.yml up -d
    docker compose --env-file .env -f compose/docker-compose.yml ps

- MinIO console: http://localhost:9001 (login from .env)
- Postgres: localhost:5432

### 2. Create the MinIO bucket

In the MinIO console, create a bucket named `warehouse` (the Iceberg warehouse root).

### 3. Download the data

    curl -L -o data/yellow_tripdata_2024-01.parquet \
      https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

~3 million January 2024 Yellow Taxi trips (~48 MB).

### 4. Load into Iceberg

    docker exec -it lakehouse-spark /opt/spark/bin/spark-submit \
      --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.apache.hadoop:hadoop-aws:3.3.4,software.amazon.awssdk:bundle:2.24.6,org.postgresql:postgresql:42.7.3 \
      --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
      /opt/spark/work-dir/jobs/load_taxi.py

## Iceberg features demonstrated

- Time travel: query the table at an earlier snapshot via VERSION AS OF <snapshot_id>
- Schema evolution: ALTER TABLE ADD COLUMN with no data-file rewrite (metadata-only)
- Partition pruning: a count filtered to one pickup_date is answered from partition metadata

## Notes / gotchas hit during setup

- MinIO image: minio/minio was removed from Docker Hub (Sept 2026); this project uses
  cgr.dev/chainguard/minio. That image is distroless (no curl/mc), so the container
  healthcheck was dropped; the /minio/health/live endpoint confirms liveness from the host.
- Postgres JDBC driver: the Iceberg JDBC catalog needs org.postgresql:postgresql in
  --packages, or connecting fails with "No suitable driver found".
- AWS region: the AWS SDK used by Iceberg's S3FileIO requires a region even against MinIO.
  AWS_REGION is set in the compose file for the Spark service.
- Ivy cache: the Spark image runs as a non-root user (home /home/spark), so the Ivy cache
  volume is mounted at /home/spark/.ivy2 and chowned to the spark user.

## Project layout

    lakehouse-ml/
    |- compose/docker-compose.yml   # MinIO + Postgres + Spark
    |- spark/load_taxi.py           # Phase 2 load job
    |- data/                        # downloaded parquet (gitignored)
    |- .env                         # credentials (gitignored)
    |- README.md
