import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, DirectoryPath

from .utils import read_yaml_credentials_file


class GeneralSettings(BaseModel):
    """Creates a Pydantic's base model for general settings.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model instance.
    """

    DATA_PATH: DirectoryPath
    RAW_FILE_NAME: str
    ARTIFACTS_PATH: DirectoryPath
    FEATURES_PATH: DirectoryPath
    TARGET_COLUMN: str
    LOG_LEVEL: str
    LOG_PATH: DirectoryPath
    RESEARCH_ENVIRONMENT_PATH: DirectoryPath


general_settings = GeneralSettings(
    **read_yaml_credentials_file(
        file_path=Path.joinpath(
            Path(__file__).resolve().parents[2],
            "config",
        ),
        file_name="settings.yaml",
    )
)

os.makedirs(general_settings.LOG_PATH, exist_ok=True)
logger.remove()
logger.add(
    Path.joinpath(general_settings.LOG_PATH, "logs", "app.log"),
    rotation="1 day",
    retention="7 days",
    compression="zip",
)
