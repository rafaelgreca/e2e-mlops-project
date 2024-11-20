"""
Loading and initializing important variables that will be used in the unit tests.
"""
from pathlib import Path

from src.config.settings import general_settings
from src.config.model import model_settings
from src.data.processing import load_dataset
from src.data.utils import load_feature
from src.model.inference import ModelServe

with open(
    f"{Path.joinpath(general_settings.RESEARCH_ENVIRONMENT_PATH, 'VERSION')}",
    "r",
    encoding="utf-8",
) as f:
    CODE_VERSION = f.readline().strip()

# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME),
    from_aws=False,
)

# # loading the processed dataset that will be used to get
# # the index of the used columns
# preprocessed_dataset = load_dataset(
#     path=Path.joinpath(
#         general_settings.DATA_PATH, f"Preprocessed_{general_settings.RAW_FILE_NAME}"
#     ),
#     from_aws=False,
# )
# FEATURES_NAME = dataset.columns.tolist()

# loading the label encoder
label_encoder = load_feature(
    path=general_settings.ARTIFACTS_PATH, feature_name="label_ohe"
)

loaded_model = ModelServe(
    model_name=model_settings.MODEL_NAME,
    model_flavor=model_settings.MODEL_FLAVOR,
    model_version=model_settings.VERSION,
)
loaded_model.load()
