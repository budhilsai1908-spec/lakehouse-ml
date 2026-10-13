# lakehouse-ml

An end-to-end lakehouse + ML serving pipeline, built to run reproducibly on
Linux/Docker. Big-data storage with Apache Iceberg, experiment tracking with
MLflow, and a live prediction API — orchestrated with Docker Compose.

**Pipeline:** NYC taxi data → Spark → Iceberg (on MinIO) → MLflow registry → prediction API

---

## Current status

- [x] **Phase 1** — Infrastructure (MinIO + Postgres on Docker)
- [x] **Phase 2** — Spark loads NYC taxi data into a partitioned Iceberg table
- [x] **Phase 3** — ML lifecycle with MLflow (tracking + model registry)
- [x] **Phase 5** — Model served as a live prediction API
- [ ] **Phase 4** — Flink streaming ingest (optional)
- [ ] **Phase 6** — Security layer (auth, TLS, least-privilege DB)

---

## Architecture

| Layer | Component | Role |
|---|---|---|
| Orchestration | Docker Compose (Linux) | Reproducible multi-service stack |
| Object storage | MinIO | S3-compatible store for Iceberg data + MLflow artifacts |
| Table format | Apache Iceberg | Partitioning, time travel, schema evolution |
| Catalog + metadata | Postgres | Iceberg JDBC catalog, MLflow backend store |
| Batch engine | Spark (PySpark) | Loads data, trains models |
| ML lifecycle | MLflow | Experiment tracking + model registry |
| Serving | FastAPI (Ray Serve target) | Loads the registered model, serves `/predict` |

---

## Prerequisites

- Docker + Docker Compose
- ~3 GB free disk for images and data

---

## Setup

### 1. Bring up the stack

```bash
docker compose --env-file .env -f compose/docker-compose.yml up -d
docker compose --env-file .env -f compose/docker-compose.yml ps
```

- MinIO console: http://localhost:9001 (login from `.env`)
- MLflow UI: http://localhost:5000
- Prediction API: http://localhost:8000
- Postgres: localhost:5432

### 2. Create the MinIO buckets

In the MinIO console, create two buckets: **`warehouse`** (Iceberg root) and
**`mlflow`** (MLflow artifacts).

### 3. Create the MLflow database

```bash
docker exec lakehouse-postgres psql -U iceberg -d catalog -c "CREATE DATABASE mlflow;"
```

### 4. Download the data

```bash
curl -L -o data/yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
```

~3 million January 2024 Yellow Taxi trips (~48 MB).

---

## Running the pipeline

### Phase 2 — Load into Iceberg

```bash
docker exec -it lakehouse-spark /opt/spark/bin/spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.apache.hadoop:hadoop-aws:3.3.4,software.amazon.awssdk:bundle:2.24.6,org.postgresql:postgresql:42.7.3 \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
  /opt/spark/work-dir/jobs/load_taxi.py
```

### Phase 3 — Train + register a model

Install the ML deps once, then train and register:

```bash
docker exec -u 0 lakehouse-spark pip install mlflow==2.17.2 scikit-learn==1.3.2 pandas boto3

docker exec -it lakehouse-spark /opt/spark/bin/spark-submit \
  --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1,org.apache.hadoop:hadoop-aws:3.3.4,software.amazon.awssdk:bundle:2.24.6,org.postgresql:postgresql:42.7.3 \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
  /opt/spark/work-dir/jobs/train.py

docker exec lakehouse-spark python3 /opt/spark/work-dir/jobs/register_best.py
```

Trains linear regression + random forest, logs both to MLflow, and registers
the lower-MAE model as `taxi-duration-model`. (Random forest wins: ~3.3 min
MAE vs ~4.8 for linear regression.)

### Phase 5 — Serve predictions

```bash
docker exec -d lakehouse-ray bash -c "cd /app && uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1"
```

Then predict:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"trip_distance":3.2,"passenger_count":1,"PULocationID":142,"DOLocationID":236,"pickup_hour":18,"pickup_dow":3}'
# -> {"predicted_minutes": 21.95}
```

---

## Iceberg features demonstrated

- **Time travel** — `SELECT COUNT(*) FROM lake.nyc.yellow_trips VERSION AS OF <snapshot_id>`
  returns the table's state at an earlier snapshot.
- **Schema evolution** — `ALTER TABLE ... ADD COLUMN` with no data-file rewrite (metadata-only, instant).
- **Partition pruning** — a count filtered to one `pickup_date` is answered from
  partition metadata; the physical plan shows a `LocalTableScan` with no data files read.

---

## Notes / gotchas hit during setup

Captured so this is reproducible and not re-debugged later:

- **MinIO image:** `minio/minio` was removed from Docker Hub (Sept 2026). This
  project uses `cgr.dev/chainguard/minio`. That image is distroless (no `curl`/`mc`),
  so the container healthcheck was dropped; `/minio/health/live` confirms liveness from the host.
- **Postgres JDBC driver:** the Iceberg JDBC catalog needs `org.postgresql:postgresql`
  in `--packages`, or connecting fails with "No suitable driver found".
- **AWS region:** the AWS SDK used by Iceberg's `S3FileIO` requires a region even
  against MinIO. `AWS_REGION` is set in the compose file for the Spark and Ray services.
- **Ivy cache:** the Spark image runs as a non-root user (home `/home/spark`), so the
  Ivy cache volume is mounted at `/home/spark/.ivy2`.
- **sklearn train/serve version match:** the model is trained and served with the same
  scikit-learn version (1.3.2). A newer serving version fails to load the saved trees
  (`monotonic_cst` AttributeError) — a classic train/serve environment-drift issue.
- **Serving layer:** `serve_model.py` (Ray Serve, with autoscaling replicas) is the
  intended production serving layer and runs on native Linux/amd64. On Apple Silicon,
  Ray's native runtime does not run reliably under amd64 emulation, so `app.py` (FastAPI)
  is used for local development. Both load the same registered model from MLflow.

---

## Project layout

```
lakehouse-ml/
├── compose/docker-compose.yml   # MinIO + Postgres + Spark + MLflow + Ray
├── spark/load_taxi.py           # Phase 2: load into Iceberg
├── ml/train.py                  # Phase 3: train + log to MLflow
├── spark/register_best.py       # Phase 3: register best model
├── serving/app.py               # Phase 5: FastAPI prediction API
├── serving/serve_model.py       # Phase 5: Ray Serve (production target)
├── data/                        # downloaded parquet (gitignored)
├── .env                         # credentials (gitignored)
└── README.md
```
