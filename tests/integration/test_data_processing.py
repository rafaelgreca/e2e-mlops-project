"""
Integration cases to test the data processing pipeline.
"""
import pathlib

import numpy as np
import pandas as pd

from src.data.processing import data_processing_inference, load_dataset
from src.config.model import model_settings
from src.config.settings import general_settings


# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=pathlib.Path.joinpath(
        general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
    ),
    from_aws=False,
)


def test_data_processing_pipeline() -> None:
    """
    Testing the integration of the entire data processing pipeline.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(columns=["id", general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=_dataset)

    assert isinstance(_dataset, pd.DataFrame)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == len(model_settings.FEATURES)
