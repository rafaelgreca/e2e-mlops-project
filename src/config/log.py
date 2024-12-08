"""
Creates a Pydantic's base model for the logging settings.
"""
import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, DirectoryPath

from . import read_yaml_credentials_file


class LoggingSettings(BaseModel):
    """Creates a Pydantic's base model for logging settings.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model instance.
    """

    LOG_LEVEL: str
    LOG_PATH: DirectoryPath


log_settings = LoggingSettings(
    **read_yaml_credentials_file(
        file_path=Path(__file__).resolve().parents[0],
        file_name="logs.yaml",
    )
)

os.makedirs(log_settings.LOG_PATH, exist_ok=True)
logger.remove()
logger.add(
    Path.joinpath(log_settings.LOG_PATH, "logs", "app.log"),
    rotation="1 day",
    retention="7 days",
    compression="zip",
)
