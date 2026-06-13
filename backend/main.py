from fastapi import FastAPI
from pydantic import BaseModel
from threading import Thread
import uuid
import time

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


app = FastAPI()
jobs = {}


class TrainRequest(BaseModel):
    features: list[str]
    train_size: float


def train_job(job_id, features, train_size):
    try:
        jobs[job_id]["status"] = "training"

        time.sleep(5)

        df = pd.read_csv("data/ecommerce_sales.csv")

        X = df[features]
        y = df["Profit"]

        categorical_columns = [
            "Region",
            "City",
            "Category",
            "Sub-Category",
            "Payment Mode"
        ]

        cat_features = [
            col for col in features
            if col in categorical_columns
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ],
            remainder="passthrough"
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "mae": round(mae, 2),
            "r2": round(r2, 3),
            "features_used": len(features),
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist()
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/train")
def start_training(request: TrainRequest):
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "created",
        "result": None,
        "error": None
    }

    thread = Thread(
        target=train_job,
        args=(job_id, request.features, request.train_size)
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "started"
    }


@app.get("/train/status/{job_id}")
def get_training_status(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}

    return jobs[job_id]