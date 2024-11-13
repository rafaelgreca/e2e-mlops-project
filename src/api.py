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
    mlflow.set_tracking_uri(f"http://127.0.0.1:5000")


@app.get("/version")
def check_versions():
    with open(
        f"{general_settings.RESEARCH_ENVIRONMENT_PATH}/VERSION", "r", encoding="utf-8"
    ) as f:
        code_version = f.readline().strip()

    return {
        "code_version": code_version,
        "model_version": model_settings.VERSION,
    }


@app.get("/predict")
async def prediction(person: Person):
    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    data = pd.DataFrame.from_dict([person.model_dump()])
    X = data_processing_inference(data)

    return {"predictions": loaded_model.predict(X).tolist()}
