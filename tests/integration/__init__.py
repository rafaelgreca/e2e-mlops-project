"""
Loading and initializing important variables that will be used in the integration tests.
"""
from pathlib import Path

from src.config.settings import general_settings
from src.data.processing import load_dataset
from src.config.model import model_settings
from src.model.inference import ModelServe

# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME),
    from_aws=False,
)

# loading the trained model
loaded_model = ModelServe(
    model_name=model_settings.MODEL_NAME,
    model_flavor=model_settings.MODEL_FLAVOR,
    model_version=model_settings.VERSION,
)
loaded_model.load()
