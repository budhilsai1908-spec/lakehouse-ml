"""
register_best.py - Phase 3: promote the best run into the MLflow Model Registry.

Finds the run with the lowest mae_minutes in the taxi-trip-duration experiment
and registers its model as 'taxi-duration-model'. Phase 5 serving pulls from here.
"""

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT = "taxi-trip-duration"
MODEL_NAME = "taxi-duration-model"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.mae_minutes ASC"],
    max_results=1,
)
best = runs[0]
mae = best.data.metrics["mae_minutes"]
model_type = best.data.params["model_type"]
print(f"Best run: {model_type}  (MAE={mae:.2f} min, run_id={best.info.run_id})")

model_uri = f"runs:/{best.info.run_id}/model"
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"Registered '{MODEL_NAME}' version {result.version}")
