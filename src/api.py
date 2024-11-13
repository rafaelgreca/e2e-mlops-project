"""
API's main file.
"""
from typing import Dict

import pandas as pd
import mlflow
from fastapi import FastAPI

from .data.processing import data_processing_inference
from .config.model import model_settings
from .config.aws import aws_credentials
from .config.settings import general_settings
from .model.inference import ModelServe
from .schema.person import Person

app = FastAPI()

if aws_credentials.EC2 != "YOUR_EC2_INSTANCE_URL":
    mlflow.set_tracking_uri(f"http://{aws_credentials.EC2}:5000")
else:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


@app.get("/version")
def check_versions() -> Dict:
    """
    This endpoint will return the current model and code versions.

    Returns:
        Dict: the model and code versions.
    """
    with open(
        f"{general_settings.RESEARCH_ENVIRONMENT_PATH}/VERSION", "r", encoding="utf-8"
    ) as file:
        code_version = file.readline().strip()

    return {
        "code_version": code_version,
        "model_version": model_settings.VERSION,
    }


@app.get("/predict")
async def prediction(person: Person) -> Dict:
    """
    This endpoint is used to make a prediction (with the trained model)
    with the given data.

    Args:
        person (Person): a person's data.

    Returns:
        Dict: the predictions.
    """
    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    data = pd.DataFrame.from_dict([person.model_dump()])
    features = data_processing_inference(data)

    return {"predictions": loaded_model.predict(features).tolist()}
