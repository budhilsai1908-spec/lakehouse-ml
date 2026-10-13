"""
app.py - Phase 5: serve the registered model with FastAPI.

Loads 'taxi-duration-model' (latest version) from the MLflow registry at
startup and exposes /health and /predict endpoints.

Start inside the container:
    uvicorn app:app --host 0.0.0.0 --port 8000

Predict:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"trip_distance":3.2,"passenger_count":1,"PULocationID":142,"DOLocationID":236,"pickup_hour":18,"pickup_dow":3}'

Note: Ray Serve (see serve_model.py) is the intended production serving layer;
it runs on native Linux/amd64. FastAPI is used here for local development on
Apple Silicon, where Ray's native runtime does not run under emulation.
"""

import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = "taxi-duration-model"
FEATURES = ["trip_distance", "passenger_count", "PULocationID",
            "DOLocationID", "pickup_hour", "pickup_dow"]

app = FastAPI(title="Taxi Trip Duration API")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")


class Trip(BaseModel):
    trip_distance: float
    passenger_count: int
    PULocationID: int
    DOLocationID: int
    pickup_hour: int
    pickup_dow: int


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(trip: Trip):
    row = pd.DataFrame([trip.model_dump()])[FEATURES]
    pred = float(model.predict(row)[0])
    return {"predicted_minutes": round(pred, 2)}
