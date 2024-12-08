"""
Unit test cases to test the model functions code.
"""
# import numpy as np
import pandas as pd

# from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

from src.config.model import model_settings
from src.data.processing import data_processing_inference
from src.model.inference import ModelServe
from .. import loaded_model


def test_load_model() -> None:
    """
    Unit case to test loading a trained model from MLflow.
    """
    assert loaded_model.model is not None

    if model_settings.MODEL_FLAVOR == "xgboost":
        assert isinstance(loaded_model.model, LGBMClassifier)

    assert isinstance(loaded_model, ModelServe)


def test_prediction() -> None:
    """
    Unit case to test making a prediction with the loaded model.
    """
    data = {
        "Age": 24.443011,
        "Height": 1.699998,
        "Weight": 81.66995,
        "Gender": "Male",
        "family_history_with_overweight": "yes",
        "CALC": "Sometimes",
        "MTRANS": "Public_Transportation",
        "FAVC": "yes",
        "FCVC": 2,
        "NCP": 2.983297,
        "CH2O": 2.763573,
        "FAF": 0,
        "TUE": 1,
        "CAEC": "Sometimes",
        "SCC": "no",
    }
    correct_prediction = "Overweight_Level_II"

    data = pd.DataFrame.from_dict([data])
    features = data_processing_inference(data)
    prediction = loaded_model.predict(features).tolist()[0]

    assert isinstance(prediction, str)
    assert prediction == correct_prediction
