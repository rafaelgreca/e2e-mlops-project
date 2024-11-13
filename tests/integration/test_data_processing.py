import pathlib

import pandas as pd
import numpy as np

from src.config.settings import general_settings
from src.config.model import model_settings
from src.data.processing import data_processing_inference, load_dataset


# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=pathlib.Path.joinpath(
        general_settings.DATA_PATH,
        general_settings.RAW_FILE_NAME
    )
)

def test_data_processing_pipeline():
    """
    Testing the integration of the entire data processing pipeline.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(columns=general_settings.TARGET_COLUMN)

    X = data_processing_inference(dataframe=_dataset)

    assert isinstance(_dataset, pd.DataFrame)
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == len(model_settings.FEATURES)
