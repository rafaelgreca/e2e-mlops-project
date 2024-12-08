"""
Loading and initializing important variables that will be used in tests.
"""
from pathlib import Path

from loguru import logger

from src.config.aws import aws_credentials
from src.config.settings import general_settings
from src.data.processing import load_dataset
from src.data.utils import download_dataset
from src.config.model import model_settings
from src.model.inference import ModelServe

use_aws = bool(aws_credentials.S3 != "YOUR_S3_BUCKET_URL")

# loading the raw dataset that was used to train the model
if Path.exists(
    Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME)
):
    logger.info(f"Downloading the dataset {general_settings.RAW_FILE_NAME}.")
    download_dataset(
        name="playground-series-s4e2",
        new_name=general_settings.RAW_FILE_NAME,
        path=general_settings.DATA_PATH,
        send_to_aws=use_aws,
        file_type="raw",
    )

logger.info(f"Loading the dataset {general_settings.RAW_FILE_NAME}.")
dataset = load_dataset(
    path=Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME),
    from_aws=use_aws,
)

logger.info(f"Loading the trained model {model_settings.MODEL_NAME}.")
loaded_model = ModelServe(
    model_name=model_settings.MODEL_NAME,
    model_flavor=model_settings.MODEL_FLAVOR,
    model_version=model_settings.VERSION,
)
loaded_model.load()
