"""
Unit test cases to test the data functions code.
"""
import os
import pathlib
import re

# import boto3
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

# from src.config.aws import aws_credentials
from src.config.settings import general_settings
from src.data.processing import (
    _categorize_numerical_columns,
    _change_height_units,
    _create_bmi_feature,
    _create_bsa_feature,
    _create_evemm_feature,
    _create_ibw_feature,
    _create_pal_feature,
    _drop_features,
    _encode_categorical_columns,
    _scale_numerical_columns,
    _transform_numerical_columns,
)
from src.data.utils import download_dataset, load_feature
from .. import dataset


def test_change_height_units() -> None:
    """
    Unit case to test the function that changes the unit measure of the
    height column.
    """
    _dataset = dataset.copy()
    max_height = max(_dataset["Height"].values.tolist())
    min_height = min(_dataset["Height"].values.tolist())

    assert max_height < 3 and min_height > 1

    _dataset = _change_height_units(dataframe=_dataset)
    max_height = max(_dataset["Height"].values.tolist())
    min_height = min(_dataset["Height"].values.tolist())

    assert max_height < 300 and min_height > 100


def test_create_bmi_feature():
    """
    Unit case to test the function that creates the Body Mass Index (BMI) feature.
    """
    _dataset = dataset.copy()

    assert "BMI" not in _dataset.columns.tolist()

    _dataset = _create_bmi_feature(dataframe=_dataset)

    assert "BMI" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["BMI"])
    assert isinstance(_dataset["BMI"].dtype, type(np.dtype("float64")))


def test_create_evemm_feature():
    """
    Unit case to test the function that creates the Eat Vegetables in Main
    Meals (EVEMM) feature.
    """
    _dataset = dataset.copy()

    assert "EVEMM" not in _dataset.columns.tolist()

    _dataset = _create_evemm_feature(dataframe=_dataset)

    assert "EVEMM" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["EVEMM"])
    assert isinstance(_dataset["EVEMM"].dtype, type(np.dtype("int64")))


def test_create_bsa_feature():
    """
    Unit case to test the function that creates the Body Surface Area (BSA) feature.
    """
    _dataset = dataset.copy()

    assert "BSA" not in _dataset.columns.tolist()

    _dataset = _create_bsa_feature(dataframe=_dataset)

    assert "BSA" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["BSA"])
    assert isinstance(_dataset["BSA"].dtype, type(np.dtype("float64")))


def test_create_ibw_feature():
    """
    Unit case to test the function that creates the Ideal Body Weight (IBW) feature.
    """
    _dataset = dataset.copy()

    assert "IBW" not in _dataset.columns.tolist()

    _dataset = _create_ibw_feature(dataframe=_dataset)

    assert "IBW" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["IBW"])
    assert isinstance(_dataset["IBW"].dtype, type(np.dtype("float64")))


def test_create_pal_feature():
    """
    Unit case to test the function that creates the Physical Activity Level (PAL) feature.
    """
    _dataset = dataset.copy()

    assert "PAL" not in _dataset.columns.tolist()

    _dataset = _create_pal_feature(dataframe=_dataset)

    assert "PAL" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["PAL"])
    assert isinstance(_dataset["PAL"].dtype, type(np.dtype("float64")))


def test_categorize_numerical_columns():
    """
    Unit case to test the function that categorizes (transform numeric to object)
    the numerical columns.
    """
    age_bins = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="qcut_bins"
    )
    _dataset = dataset.copy()
    _dataset = _create_evemm_feature(dataframe=_dataset)

    assert isinstance(_dataset["Age"].dtype, type(np.dtype("float64")))
    assert isinstance(_dataset["EVEMM"].dtype, type(np.dtype("int64")))

    _dataset = _categorize_numerical_columns(
        dataframe=_dataset,
        bins=age_bins,
    )

    assert isinstance(_dataset["Age"].dtype, type(np.dtype("object")))
    assert isinstance(_dataset["EVEMM"].dtype, type(np.dtype("object")))


def test_scale_numerical_columns():
    """
    Unit case to test the function that scales the numerical features.
    """
    _dataset = dataset.copy()

    _dataset = _change_height_units(_dataset)

    # Creating the BMI feature
    _dataset = _create_bmi_feature(_dataset)

    # Creating the PAL feature
    _dataset = _create_pal_feature(_dataset)

    # Creating the BSA feature
    _dataset = _create_bsa_feature(_dataset)

    # Creating the IBW feature
    _dataset = _create_ibw_feature(_dataset)

    _dataset = _drop_features(dataframe=_dataset, features=["id", "Age"])

    scalers = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="features_sc"
    )
    numerical_columns = _dataset.select_dtypes(exclude="object").columns.tolist()

    _dataset = _transform_numerical_columns(dataframe=_dataset)
    _dataset2 = _scale_numerical_columns(dataframe=_dataset, scalers=scalers)

    for column in numerical_columns:
        assert round(_dataset2[column].mean(axis=0), 2) == 0
        assert round(_dataset2[column].std(axis=0), 2) == 1
        assert _dataset[column].mean(axis=0) >= _dataset2[column].mean(axis=0)
        assert _dataset[column].var(axis=0) >= _dataset2[column].var(axis=0)
        assert _dataset[column].std(axis=0) >= _dataset2[column].std(axis=0)
        assert isinstance(_dataset2[column].dtype, type(np.dtype("float64")))


def test_transform_numerical_columns():
    """
    Unit case to test the function that applies log transformatio to the
    numerical columns.
    """
    _dataset = dataset.copy()

    _dataset = _change_height_units(_dataset)

    # Creating the BMI feature
    _dataset = _create_bmi_feature(_dataset)

    # Creating the BSA feature
    _dataset = _create_bsa_feature(_dataset)

    # Creating the IBW feature
    _dataset = _create_ibw_feature(_dataset)

    _dataset = _drop_features(dataframe=_dataset, features=["id", "Age"])

    _dataset2 = _transform_numerical_columns(dataframe=_dataset)

    numerical_columns = _dataset2.select_dtypes(exclude="object").columns.tolist()

    for column in numerical_columns:
        assert _dataset[column].mean(axis=0) != _dataset2[column].mean(axis=0)
        assert _dataset[column].var(axis=0) != _dataset2[column].var(axis=0)
        assert _dataset[column].std(axis=0) != _dataset2[column].std(axis=0)
        assert _dataset[column].max(axis=0) != _dataset2[column].max(axis=0)
        assert _dataset[column].min(axis=0) != _dataset2[column].min(axis=0)


def test_encode_categorical_columns():
    """
    Unit case to test the function that encodes (applies the one hot
    encode technique) the categorical features.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(columns=["NObeyesdad"])  # removing the target column
    categorical_columns = _dataset.select_dtypes(include="object").columns.tolist()

    encoders = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="features_ohe"
    )
    _dataset2 = _encode_categorical_columns(dataframe=_dataset, encoders=encoders)

    for column in categorical_columns:
        assert (
            column in _dataset.columns.tolist()
            and column not in _dataset2.columns.tolist()
        )
        assert any(re.findall(f"{column}_", c) for c in _dataset2.columns.tolist())
        assert _dataset.shape[1] != _dataset2.shape[1]


def test_load_dataset():
    """
    Unit case to test the function that loads the original, raw dataset.
    """
    columns = [
        "id",
        "Gender",
        "Age",
        "Height",
        "Weight",
        "family_history_with_overweight",
        "FAVC",
        "FCVC",
        "NCP",
        "CAEC",
        "SMOKE",
        "CH2O",
        "SCC",
        "FAF",
        "TUE",
        "CALC",
        "MTRANS",
        "NObeyesdad",
    ]

    assert isinstance(dataset, pd.DataFrame)
    assert all(c in dataset.columns.tolist() for c in columns)
    assert dataset.shape[1] == len(columns)
    assert dataset.shape[0] == 20758


def test_download_current_dataset():
    """
    Unit case to test the function that downloads the current dataset.
    """
    if pathlib.Path.exists(
        pathlib.Path.joinpath(
            general_settings.DATA_PATH, general_settings.CURRENT_FILE_NAME
        )
    ):
        os.remove(
            pathlib.Path.joinpath(
                general_settings.DATA_PATH, general_settings.CURRENT_FILE_NAME
            )
        )

    download_dataset(
        name="aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster",
        new_name=general_settings.CURRENT_FILE_NAME,
        path=general_settings.DATA_PATH,
        send_to_aws=False,
        file_type="current",
    )

    assert pathlib.Path.exists(
        pathlib.Path.joinpath(
            general_settings.DATA_PATH, general_settings.CURRENT_FILE_NAME
        )
    )


def test_download_raw_dataset():
    """
    Unit case to test the function that downloads the raw, original dataset.
    """
    if pathlib.Path.exists(
        pathlib.Path.joinpath(general_settings.DATA_PATH, "playground-series-s4e2")
    ):
        os.remove(
            pathlib.Path.joinpath(general_settings.DATA_PATH, "playground-series-s4e2")
        )

    download_dataset(
        name="playground-series-s4e2",
        new_name=general_settings.RAW_FILE_NAME,
        path=general_settings.DATA_PATH,
        send_to_aws=False,
        file_type="raw",
    )

    assert pathlib.Path.exists(
        pathlib.Path.joinpath(
            general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
        )
    )


# def test_send_dataset_aws():
#     """
#     Unit case to test the function that sends the original, raw dataset to
#     an AWS S3 bucket.
#     """
#     if pathlib.Path.exists(
#         pathlib.Path.joinpath(
#             general_settings.DATA_PATH,
#             general_settings.RAW_FILE_NAME
#         )
#     ):
#         os.remove(
#             pathlib.Path.joinpath(
#                 general_settings.DATA_PATH,
#                 general_settings.RAW_FILE_NAME
#             )
#         )

#     download_dataset(
#         name="aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster",
#         new_name=general_settings.RAW_FILE_NAME,
#         path=general_settings.DATA_PATH,
#         send_to_aws=True,
#         file_type="raw",
#     )

#     assert not pathlib.Path.exists(
#         pathlib.Path.joinpath(
#             general_settings.DATA_PATH,
#             general_settings.RAW_FILE_NAME
#         )
#     )

#     bucket = boto3.client(
#         "s3",
#         aws_access_key_id=aws_credentials.AWS_ACCESS_KEY,
#         aws_secret_access_key=aws_credentials.AWS_SECRET_KEY,
#     )
#     objs = list(bucket.objects.filter(Prefix=general_settings.RAW_FILE_NAME))

#     assert len(objs) > 0
#     assert any(general_settings.RAW_FILE_NAME in str(o) for o in objs)
