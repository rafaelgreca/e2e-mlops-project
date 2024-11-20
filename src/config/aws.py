"""
Creates a Pydantic's base model for the AWS' credentials.
"""
from pathlib import Path

from pydantic import BaseModel

from . import read_yaml_credentials_file


class AWSCredentials(BaseModel):
    """Creates a Pydantic's base model for the AWS credentials.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model instance.
    """

    EC2: str
    S3: str
    POSTGRESQL: str
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str


aws_credentials = AWSCredentials(
    **read_yaml_credentials_file(
        file_path=Path(__file__).resolve().parents[0],
        file_name="credentials.yaml",
    )
)
