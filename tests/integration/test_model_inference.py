"""
Integration cases to test the model inference pipeline.
"""
import numpy as np
import pandas as pd

from src.config.model import model_settings
from src.config.settings import general_settings
from src.data.processing import data_processing_inference
from .. import dataset, loaded_model


def test_model_inference_pipeline() -> None:
    """
    Testing the integration of the entire model inference pipeline.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(columns=["id", general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=_dataset)

    assert isinstance(_dataset, pd.DataFrame)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == len(model_settings.FEATURES)

    assert loaded_model.model is not None

    predictions = loaded_model.predict(features, transform_to_str=False)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == features.shape[0]
    assert isinstance(predictions.dtype, type(np.dtype("int64")))

    predictions = loaded_model.predict(features, transform_to_str=True)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == features.shape[0]

    predictions = predictions.tolist()
    assert isinstance(predictions[0], str)
