import pathlib

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.model.inference import ModelServe
from src.config.model import model_settings
from src.config.settings import general_settings
from src.data.utils import load_feature
from src.data.processing import data_processing_inference, load_dataset

# loading the label encoder
label_encoder = load_feature(
    path=general_settings.ARTIFACTS_PATH,
    feature_name='label_ohe'
)

# loading the processed dataset that will be used to get
# the index of the used columns
dataset = load_dataset(
    path=pathlib.Path.joinpath(
        general_settings.DATA_PATH,
        "Preprocessed_ObesityDataSet.csv"
    )
)
FEATURES_NAME = dataset.columns.tolist()

def test_load_model() -> None:
    """
    Unit case to test loading a trained model from MLflow.
    """
    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    assert loaded_model.model is not None

    if model_settings.MODEL_FLAVOR == "xgboost":
        assert isinstance(loaded_model.model, XGBClassifier)

    assert isinstance(loaded_model, ModelServe)

def test_prediction() -> None:
    """
    Unit case to test making a prediction with the loaded model.
    """
    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    data = {
        "Age": 21,
        "CAEC": "Sometimes",
        "CALC": "no",
        "FAF": 0,
        "FCVC": 2,
        "Gender": "Female",
        "Height": 1.62,
        "MTRANS": "Public_Transportation",
        "SCC": "no",
        "SMOKE": "False",
        "TUE": 1,
        "Weight": 64
    }
    correct_prediction = "Normal_Weight"

    data = pd.DataFrame.from_dict([data])
    X = data_processing_inference(data)
    prediction = loaded_model.predict(X).tolist()[0][0]

    assert isinstance(prediction, str)
    assert prediction == correct_prediction

def test_model_performance() -> None:
    """
    Unit case to test the model performance on training and validation sets
    (making sure that are the same values as mentioned in MLflow's UI).
    """
    # FIXME: fix this
    indexes = [FEATURES_NAME.index(i) for i in model_settings.FEATURES]

    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    X_train = load_feature(
        path=general_settings.FEATURES_PATH,
        feature_name='X_train'
    )[:, indexes]
    y_train = load_feature(
        path=general_settings.FEATURES_PATH,
        feature_name='y_train'
    )
    y_train = np.max(y_train, axis=1)

    train_predictions = loaded_model.predict(X_train, transform_to_str=False)
    train_score = f1_score(y_true=y_train, y_pred=train_predictions, average="weighted")

    X_valid = load_feature(
        path=general_settings.FEATURES_PATH,
        feature_name='X_valid'
    )[:, indexes]
    y_valid = load_feature(
        path=general_settings.FEATURES_PATH,
        feature_name='y_valid'
    )
    y_valid = np.max(y_valid, axis=1)

    valid_predictions = loaded_model.predict(X_valid, transform_to_str=False)
    valid_score = f1_score(y_true=y_valid, y_pred=valid_predictions, average="weighted")

    assert train_score == train_score
    assert valid_score == valid_score
