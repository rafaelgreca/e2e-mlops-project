"""
Loading and initializing important variables that will be used in the unit tests.
"""
from pathlib import Path

from src.config.aws import aws_credentials
from src.config.settings import general_settings

use_aws = bool(aws_credentials.S3 != "YOUR_S3_BUCKET_URL")

with open(
    f"{Path.joinpath(general_settings.RESEARCH_ENVIRONMENT_PATH, 'VERSION')}",
    "r",
    encoding="utf-8",
) as f:
    CODE_VERSION = f.readline().strip()

# # loading the processed dataset that will be used to get
# # the index of the used columns
# preprocessed_dataset = load_dataset(
#     path=Path.joinpath(
#         general_settings.DATA_PATH, f"Preprocessed_{general_settings.RAW_FILE_NAME}"
#     ),
#     from_aws=use_aws,
# )
# FEATURES_NAME = dataset.columns.tolist()
