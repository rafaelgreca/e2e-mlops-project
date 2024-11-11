import pandas as pd
from fastapi import FastAPI

from .data.processing import load_dataset, data_processing_inference
from .config.settings import general_settings
from .config.model import model_settings
from .model.inference import ModelServe
from .schema.person import Person
from .schema.prediction import Prediction


app = FastAPI()

@app.get("/version")
def check_versions():
    with open("../notebooks/VERSION", "r", encoding="utf-8") as f:
        code_version = f.readline().strip()

    return {
        "code_version": code_version,
        "model_version": model_settings.VERSION,
    }

@app.get("/predict")
async def prediction(person: Person, response_model=Prediction):
    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()
    
    data = pd.DataFrame.from_dict([person.model_dump()])
    X = data_processing_inference(data)

    return {
        "predictions": loaded_model.predict(X).tolist()
    }