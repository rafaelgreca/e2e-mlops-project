"""
Unit test cases to test the model functions code.
"""
import pathlib
from os import PathLike
from typing import List

from src.config.aws import aws_credentials
from src.config.kaggle import kaggle_credentials
from src.config.model import model_settings
from src.config.settings import general_settings
from src.config.log import log_settings


def test_aws_file() -> None:
    """
    Unit case to test the function responsible for reading an YAML
    file and the content of the 'aws' credentials file.
    """
    assert isinstance(aws_credentials.EC2, str)
    assert isinstance(aws_credentials.S3, str)
    assert isinstance(aws_credentials.POSTGRESQL, str)
    assert isinstance(aws_credentials.AWS_ACCESS_KEY, str)
    assert isinstance(aws_credentials.AWS_SECRET_KEY, str)


def test_kaggle_file() -> None:
    """
    Unit case to test the function responsible for reading an YAML
    file and the content of the 'kaggle' credentials file.
    """
    assert isinstance(kaggle_credentials.KAGGLE_USERNAME, str)
    assert isinstance(kaggle_credentials.KAGGLE_KEY, str)


def test_model_file() -> None:
    """
    Unit case to test the function responsible for reading an YAML
    file and the content of the 'model' file.
    """
    assert isinstance(model_settings.MODEL_NAME, str)
    assert isinstance(model_settings.MODEL_FLAVOR, str)
    assert isinstance(model_settings.EXPERIMENT_ID, str)
    assert isinstance(model_settings.VERSION, str)
    assert isinstance(model_settings.RUN_ID, str)
    assert isinstance(model_settings.FEATURES, List)


def test_settings_file() -> None:
    """
    Unit case to test the function responsible for reading an YAML
    file and the content of the 'settings' file.
    """
    assert isinstance(general_settings.DATA_PATH, PathLike)
    assert pathlib.Path.exists(general_settings.DATA_PATH)
    assert isinstance(general_settings.RAW_FILE_NAME, str)
    assert isinstance(general_settings.CURRENT_FILE_NAME, str)
    assert isinstance(general_settings.ARTIFACTS_PATH, PathLike)
    assert pathlib.Path.exists(general_settings.ARTIFACTS_PATH)
    assert isinstance(general_settings.FEATURES_PATH, PathLike)
    assert pathlib.Path.exists(general_settings.ARTIFACTS_PATH)
    assert isinstance(general_settings.TARGET_COLUMN, str)
    assert isinstance(log_settings.LOG_LEVEL, str)
    assert isinstance(log_settings.LOG_PATH, PathLike)
    assert pathlib.Path.exists(log_settings.LOG_PATH)
    assert isinstance(general_settings.RESEARCH_ENVIRONMENT_PATH, PathLike)
    assert pathlib.Path.exists(general_settings.RESEARCH_ENVIRONMENT_PATH)
