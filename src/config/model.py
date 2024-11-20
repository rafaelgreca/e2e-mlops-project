"""
Creates a Pydantic's base model for the model's configuration.
"""
from typing import List

from pathlib import Path
from pydantic import BaseModel

from . import read_yaml_credentials_file


class ModelSettings(BaseModel):
    """Creates a Pydantic's base model for the model settings.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model instance.
    """

    MODEL_NAME: str
    VERSION: str
    MODEL_FLAVOR: str
    EXPERIMENT_ID: str
    RUN_ID: str
    FEATURES: List[str]


model_settings = ModelSettings(
    **read_yaml_credentials_file(
        file_path=Path(__file__).resolve().parents[0],
        file_name="model.yaml",
    )
)
