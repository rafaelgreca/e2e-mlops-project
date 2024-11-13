import pathlib
import re
import os
from typing import List

import boto3
import pytest
import pandas.api.types as ptypes
import pandas as pd
import numpy as np

from src.data.processing import (
    _drop_features,
    _change_height_units,
    load_dataset,
    _remove_duplicates,
    _remove_outliers,
    _create_is_feature,
    _create_bmi_feature,
    _create_bmr_feature,
    _categorize_numerical_columns,
    _scale_numerical_columns,
    _transform_numerical_columns,
    _encode_categorical_columns,
)
from src.config.settings import general_settings
from src.config.aws import aws_credentials
from src.data.utils import load_feature, download_dataset

# loading the raw dataset that was used to train the model
dataset = load_dataset(
    path=pathlib.Path.joinpath(
        general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
    )
)


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


@pytest.mark.parametrize(
    "features",
    [
        ["Height", "Weight", "Gender", "Age"],
        ["family_history_with_overweight", "FAVC", "FCVC", "NCP"],
        ["CAEC", "SMOKE", "CH2O", "SCC"],
        ["FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"],
    ],
)
def test_drop_features(features: List[str]) -> None:
    """
    Unit case to test the function that drops a certain set of features
    from the dataset.
    """
    _dataset = dataset.copy()

    assert all(f in _dataset.columns.tolist() for f in features)

    _dataset = _drop_features(dataframe=_dataset, features=features)

    assert all(f not in _dataset.columns.tolist() for f in features)


def test_remove_duplicates():
    """
    Unit case to test the function that drops duplicated rows from the dataset.
    """
    _dataset = dataset.copy()
    shape_before = _dataset.shape[0]

    _dataset = _remove_duplicates(dataframe=_dataset)
    shape_after = _dataset.shape[0]

    assert shape_after < shape_before


def test_remove_outliers():
    """
    Unit case to test the function that removes outliers from the dataset
    based on the person's age.
    """
    _dataset = dataset.copy()
    shape_before = _dataset.shape[0]

    _dataset = _remove_outliers(dataframe=_dataset)
    shape_after = _dataset.shape[0]

    assert shape_after < shape_before


def test_create_is_feature():
    """
    Unit case to test the function that creates the Is Sedentary (IS) feature.
    """
    _dataset = dataset.copy()

    assert "IS" not in _dataset.columns.tolist()

    _dataset = _create_is_feature(dataframe=_dataset)

    assert "IS" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["IS"])
    assert isinstance(_dataset["IS"].dtype, type(np.dtype("int64")))
    assert max(_dataset["IS"].values.tolist()) <= 1


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


def test_create_bmr_feature():
    """
    Unit case to test the function that creates the Body Mass Ratio (BMR) feature.
    """
    _dataset = dataset.copy()

    assert "BMR" not in _dataset.columns.tolist()

    _dataset = _create_bmr_feature(dataframe=_dataset)

    assert "BMR" in _dataset.columns.tolist()
    assert ptypes.is_numeric_dtype(_dataset["BMR"])
    assert isinstance(_dataset["BMR"].dtype, type(np.dtype("float64")))


def test_categorize_numerical_columns():
    """
    Unit case to test the function that categorizes (transform numeric to object)
    the numerical columns.
    """
    age_bins = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="qcut_bins"
    )
    _dataset = dataset.copy()
    _dataset = _create_is_feature(dataframe=_dataset)

    assert isinstance(_dataset["Age"].dtype, type(np.dtype("float64")))
    assert isinstance(_dataset["IS"].dtype, type(np.dtype("int64")))

    _dataset = _categorize_numerical_columns(
        dataframe=_dataset,
        bins=age_bins,
    )

    assert isinstance(_dataset["Age"].dtype, type(np.dtype("object")))
    assert isinstance(_dataset["IS"].dtype, type(np.dtype("object")))


def test_scale_numerical_columns():
    """
    Unit case to test the function that scales the numerical features.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(
        columns=["Age"]
    )  # this column will be transformed to object
    _dataset = _dataset.drop(
        columns=["Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    )  # these columns are not being used

    numerical_columns = _dataset.select_dtypes(exclude="object").columns.tolist()

    sc = load_feature(path=general_settings.ARTIFACTS_PATH, feature_name="features_sc")
    _dataset2 = _scale_numerical_columns(dataframe=_dataset, sc=sc)

    for nc in numerical_columns:
        assert _dataset2[nc].mean(axis=0) == 0
        assert _dataset2[nc].std(axis=0) == 1
        assert _dataset[nc].mean(axis=0) > _dataset2[nc].mean(axis=0)
        assert _dataset[nc].var(axis=0) > _dataset2[nc].var(axis=0)
        assert _dataset[nc].std(axis=0) > _dataset2[nc].std(axis=0)
        assert isinstance(_dataset2[nc].dtype, type(np.dtype("float64")))


def test_transform_numerical_columns():
    """
    Unit case to test the function that applies log transformatio to the
    numerical columns.
    """
    _dataset = dataset.copy()
    _dataset = _dataset.drop(
        columns=["Age"]
    )  # this column will be transformed to object
    _dataset = _dataset.drop(
        columns=["Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    )  # these columns are not being used
    numerical_columns = _dataset.select_dtypes(exclude="object").columns.tolist()

    _dataset2 = _transform_numerical_columns(dataframe=_dataset)

    for nc in numerical_columns:
        assert _dataset[nc].mean(axis=0) != _dataset2[nc].mean(axis=0)
        assert _dataset[nc].var(axis=0) != _dataset2[nc].var(axis=0)
        assert _dataset[nc].std(axis=0) != _dataset2[nc].std(axis=0)
        assert _dataset[nc].max(axis=0) != _dataset2[nc].max(axis=0)
        assert _dataset[nc].min(axis=0) != _dataset2[nc].min(axis=0)


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
    _dataset2 = _encode_categorical_columns(
        dataframe=_dataset, encoders=encoders, target_column="NObeyesdad"
    )

    for cc in categorical_columns:
        assert cc in _dataset.columns.tolist() and cc not in _dataset2.columns.tolist()
        assert any(re.findall(f"{cc}_", c) for c in _dataset2.columns.tolist())
        assert _dataset.shape[1] != _dataset2.shape[1]


def test_load_dataset():
    """
    Unit case to test the function that loads the original, raw dataset.
    """
    columns = [
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
    assert dataset.shape[0] == 2111


def test_download_dataset():
    """
    Unit case to test the function that downloads the original, raw dataset.
    """
    if pathlib.Path.exists(
        pathlib.Path.joinpath(
            general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
        )
    ):
        os.remove(
            pathlib.Path.joinpath(
                general_settings.DATA_PATH, general_settings.RAW_FILE_NAME
            )
        )

    download_dataset(
        name="aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster",
        new_name=general_settings.RAW_FILE_NAME,
        path=general_settings.DATA_PATH,
        send_to_aws=False,
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
