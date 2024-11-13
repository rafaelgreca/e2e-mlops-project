"""
Integration cases to test the model inference pipeline.
"""
import pathlib

import numpy as np
import pandas as pd

from src.config.model import model_settings
from src.config.settings import general_settings
from src.data.processing import data_processing_inference, load_dataset
from src.model.inference import ModelServe

# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=pathlib.Path.joinpath(
        general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
    )
)


def test_model_inference_pipeline() -> None:
    """
    Testing the integration of the entire model inference pipeline.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(columns=general_settings.TARGET_COLUMN)

    features = data_processing_inference(dataframe=_dataset)

    assert isinstance(_dataset, pd.DataFrame)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == len(model_settings.FEATURES)

    loaded_model = ModelServe(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()

    assert loaded_model.model is not None

    predictions = loaded_model.predict(features, transform_to_str=False)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == features.shape[0]
    assert isinstance(predictions.dtype, type(np.dtype("float64")))

    # FIXME: fix this
    # predictions = loaded_model.predict(x, transform_to_str=True)

    # assert isinstance(predictions, List)
    # assert len(predictions) == x.shape[0]
    # assert isinstance(type(predictions[0]), str)
