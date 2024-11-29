"""
Loading and initializing important variables that will be used in the api code.
"""
from pathlib import Path

import mlflow
from loguru import logger

from ..data.processing import load_dataset
from ..data.utils import download_dataset
from ..config.aws import aws_credentials
from ..config.model import model_settings
from ..config.settings import general_settings
from ..model.inference import ModelServe

use_aws = bool(aws_credentials.S3 != "YOUR_S3_BUCKET_URL")

if aws_credentials.EC2 != "YOUR_EC2_INSTANCE_URL":
    mlflow.set_tracking_uri(f"http://{aws_credentials.EC2}:5000")
else:
    mlflow.set_tracking_uri("http://mlflow:5000")

if not Path.exists(
    Path.joinpath(general_settings.DATA_PATH, general_settings.CURRENT_FILE_NAME)
):
    logger.info(f"Downloading the {general_settings.CURRENT_FILE_NAME} dataset.")

    download_dataset(
        name="aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster",
        new_name=general_settings.CURRENT_FILE_NAME,
        path=general_settings.DATA_PATH,
        send_to_aws=use_aws,
        file_type="current",
    )

logger.info(f"Loading the {general_settings.CURRENT_FILE_NAME} dataset.")
current_dataset = load_dataset(
    path=Path.joinpath(general_settings.DATA_PATH, general_settings.CURRENT_FILE_NAME),
    from_aws=use_aws,
)

logger.info(f"Loading {model_settings.MODEL_NAME} pre-trained model.")
loaded_model = ModelServe(
    model_name=model_settings.MODEL_NAME,
    model_flavor=model_settings.MODEL_FLAVOR,
    model_version=model_settings.VERSION,
)
loaded_model.load()

logger.info("Loading the reference data and filtering its columns.")
reference_data = load_dataset(
    path=Path.joinpath(
        general_settings.DATA_PATH, f"Preprocessed_{general_settings.RAW_FILE_NAME}"
    ),
    from_aws=use_aws,
)
reference_data = reference_data[
    model_settings.FEATURES + [general_settings.TARGET_COLUMN]
]
reference_data["prediction"] = loaded_model.predict(
    reference_data[model_settings.FEATURES].values
)
