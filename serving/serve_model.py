"""
serve_model.py - Phase 5: serve the registered model with Ray Serve.

Loads 'taxi-duration-model' (latest version) from the MLflow registry and
exposes a /predict HTTP endpoint. Runs with autoscaling replicas (1-3).

Start inside the Ray container:
    serve run serve_model:entrypoint

Then POST to http://localhost:8000/predict with a JSON body:
    {"trip_distance": 3.2, "passenger_count": 1, "PULocationID": 142,
     "DOLocationID": 236, "pickup_hour": 18, "pickup_dow": 3}
"""

import os
import mlflow
import pandas as pd
from ray import serve
from starlette.requests import Request

MODEL_NAME = "taxi-duration-model"
FEATURES = ["trip_distance", "passenger_count", "PULocationID",
            "DOLocationID", "pickup_hour", "pickup_dow"]


@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 3,
                        "target_ongoing_requests": 5},
    ray_actor_options={"num_cpus": 1},
)
class DurationModel:
    def __init__(self):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        # Load the latest registered version of the model.
        self.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")

    async def __call__(self, request: Request):
        body = await request.json()
        row = pd.DataFrame([{f: body.get(f) for f in FEATURES}])
        pred = float(self.model.predict(row)[0])
        return {"predicted_minutes": round(pred, 2)}


entrypoint = DurationModel.bind()
