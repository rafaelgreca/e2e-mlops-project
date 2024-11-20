"""
Creates a Pydantic's base model for the reports settings.
"""
import os
from pathlib import Path

from pydantic import BaseModel, DirectoryPath

from . import read_yaml_credentials_file


class ReportSettings(BaseModel):
    """Creates a Pydantic's base model for report settings.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model instance.
    """

    REPORTS_PATH: DirectoryPath
    TARGET_DRIFT_REPORT_NAME: str
    DATA_DRIFT_REPORT_NAME: str
    DATA_QUALITY_REPORT_NAME: str
    MODEL_PERFORMANCE_REPORT_NAME: str


report_settings = ReportSettings(
    **read_yaml_credentials_file(
        file_path=Path(__file__).resolve().parents[0],
        file_name="reports.yaml",
    )
)

os.makedirs(report_settings.REPORTS_PATH, exist_ok=True)
