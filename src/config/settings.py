from pathlib import Path
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

general_settings = GeneralSettings(
    **read_yaml_credentials_file(
        file_path=Path.joinpath(
            Path(__file__).resolve().parents[2],
            "config",
        ),
        file_name="settings.yaml",
    )
)
