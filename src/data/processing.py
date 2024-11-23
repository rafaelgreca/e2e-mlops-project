"""
Stores data processing functions, such as for cleaning the data, creating new features,
enconding categorical columns, and so on.
"""
import os
import pathlib
from typing import List, Dict

import boto3
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ..config.aws import aws_credentials
from ..config.model import model_settings
from ..config.settings import general_settings
from .utils import load_feature


def data_processing_inference(dataframe: pd.DataFrame) -> np.ndarray:
    """Applies the data processing pipeline.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        np.ndarray: the features array.
    """
    # First step) changing the height unit
    logger.info("Changing the height units to centimeters.")
    dataframe = _change_height_units(dataframe)

    # Feature engineering step)
    # Creating the BMI feature
    logger.info("Creating a new column for the BMI values from the data samples.")
    dataframe = _create_bmi_feature(dataframe)

    # Creating the PAL feature
    logger.info("Creating a new column for the PAL values from the data samples.")
    dataframe = _create_pal_feature(dataframe)

    # Creating the BSA feature
    logger.info("Creating a new column for the BSA values from the data samples.")
    dataframe = _create_bsa_feature(dataframe)

    # Creating the IBW feature
    logger.info("Creating a new column for the IBW values from the data samples.")
    dataframe = _create_ibw_feature(dataframe)

    # Creating the EVEMM feature
    logger.info("Creating a new column for the EVEMM values from the data samples.")
    dataframe = _create_evemm_feature(dataframe)

    # Feature transformation step)
    # Transforming the AGE and EVEMM columns in categorical
    logger.info("Categorizing the numerical columns 'Age' and 'EVEMM'.")
    age_bins = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="qcut_bins"
    )
    dataframe = _categorize_numerical_columns(dataframe, age_bins)

    # Transforming (Log Transformation) numerical columns
    dataframe = _transform_numerical_columns(dataframe)

    # Loading the encoders and scalers
    logger.info(
        f"Loading encoders 'features_ohe' from path {general_settings.ARTIFACTS_PATH}."
    )
    encoders = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="features_ohe"
    )

    logger.info(
        f"Loading scalers 'features_sc' from path {general_settings.ARTIFACTS_PATH}."
    )
    scalers = load_feature(
        path=general_settings.ARTIFACTS_PATH, feature_name="features_sc"
    )

    # Scaling numerical columns
    dataframe = _scale_numerical_columns(dataframe=dataframe, scalers=scalers)

    # Encoding categorical columns
    dataframe = _encode_categorical_columns(
        dataframe=dataframe,
        encoders=encoders,
    )

    # Selecting only the features that are important for the model
    dataframe = dataframe[model_settings.FEATURES]
    logger.info(
        f"Filtering the features columns, keeping only {model_settings.FEATURES} columns."
    )

    features = dataframe.values
    return features


def _change_height_units(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Changes the Height unit to centimeters, so will be easier to calculate
    other features from it.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with the 'height' column transformed.
    """
    dataframe["Height"] *= 100
    return dataframe


def _create_bmi_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Body Mass Index (BMI) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of BMI for each data.
    """
    dataframe["BMI"] = dataframe["Weight"] / (dataframe["Height"] ** 2)
    return dataframe


def _create_pal_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Physical Activity Level (PAL) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of PAL for each data.
    """
    dataframe["PAL"] = dataframe["FAF"] - dataframe["TUE"]
    return dataframe


def _create_evemm_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Eat Vegetables Every Main Meal (EVEMM) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of EVEMM for each data.
    """
    dataframe["EVEMM"] = dataframe["FCVC"] >= dataframe["NCP"]
    dataframe["EVEMM"] = dataframe["EVEMM"].astype(int)
    return dataframe


def _create_bsa_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Body Surface Area (BSA) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of BSA for each data.
    """

    def _calculate_bsa(gender: str, height: float, weight: float) -> float:
        """
        Auxiliary function that calculates the BSA based on the Schlich's formula.

        Args:
            gender (str): the person's gender.
            height (float): the person's height.
            weight (float): the person's weight.

        Returns:
            float: the BSA value for that person.
        """
        # Schlich formula
        if gender == "Female":
            return 0.000975482 * (weight**0.46) * (height**1.08)

        return 0.000579479 * (weight**0.38) * (height**1.24)

    dataframe["BSA"] = dataframe.apply(
        lambda x: _calculate_bsa(x["Gender"], x["Height"], x["Weight"]), axis=1
    )
    return dataframe


def _create_ibw_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Ideal Body Weight (IBW) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of IBW for each data.
    """

    def calculate_ibw(gender: str, height: float) -> float:
        """
        Auxiliary function that calculates the IBW based on the B. J. Devine's formula.

        Args:
            gender (str): the person's gender.
            height (float): the person's height.

        Returns:
            float: the IBW value for that person.
        """
        # B. J. Devine formula
        if gender == "Female":
            return 45.5 + 0.9 * (height - 152)

        return 50 + 0.9 * (height - 152)

    dataframe["IBW"] = dataframe.apply(
        lambda x: calculate_ibw(x["Gender"], x["Height"]), axis=1
    )
    return dataframe


def _categorize_numerical_columns(
    dataframe: pd.DataFrame,
    bins: pd.DataFrame,
) -> pd.DataFrame:
    """Categorizes the numerical columns (e.g., transforming int to object/class).

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with all numerical columns categorized.
    """
    dataframe["Age"] = pd.cut(
        x=dataframe["Age"], bins=bins, labels=["q1", "q2", "q3", "q4"]
    )
    dataframe["Age"] = dataframe["Age"].astype("object")
    dataframe["EVEMM"] = dataframe["EVEMM"].astype("object")
    return dataframe


def _transform_numerical_columns(
    dataframe: pd.DataFrame, epsilon: float = 1e-10
) -> pd.DataFrame:
    """Transforms the numerical columns using the Log Transformation technique.

    Args:
        dataframe (pd.DataFrame): the dataframe.
        epsilon (float): a really small value (called epsilon)
            used to avoid calculate the log of 0. Defailts to 1e-10.

    Returns:
        transformed_dataframe (pd.DataFrame): the dataframe with all numerical columns transformed.
    """
    numerical_columns = dataframe.select_dtypes(exclude="object").columns.tolist()
    logger.info(f"Applying Log Transformation to the {numerical_columns} columns.")
    transformed_dataframe = dataframe.copy()

    for column in numerical_columns:
        if column != "PAL":
            transformed_dataframe[column] = np.log(dataframe[column].values + epsilon)

    return transformed_dataframe


def _scale_numerical_columns(
    dataframe: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
) -> pd.DataFrame:
    """Scales the numerical columns using the Standard technique.

    Args:
        dataframe (pd.DataFrame): the dataframe.
        scalers (Dict[str, OneHotEncoder]): a dict containing the corresponding
            encoder for each feature.

    Returns:
        pd.DataFrame: the dataframe with all numerical columns encoded.
    """
    numerical_columns = dataframe.select_dtypes(exclude="object").columns.tolist()
    logger.info(f"Scaling the {numerical_columns} columns.")

    for column in numerical_columns:
        dataframe[column] = scalers[column].transform(
            dataframe[column].values.reshape(-1, 1)
        )

    return dataframe


def _encode_categorical_columns(
    dataframe: pd.DataFrame,
    encoders: Dict[str, OneHotEncoder],
) -> pd.DataFrame:
    """Encodes the categorical columns using the OneHot technique.

    Args:
        dataframe (pd.DataFrame): the dataframe.
        encoders (Dict[str, OneHotEncoder]): a dict containing the corresponding
            encoder for each feature.

    Returns:
        pd.DataFrame: the dataframe with all categorical columns encoded.
    """
    categorical_columns = dataframe.select_dtypes(include="object").columns.tolist()
    logger.info(f"Encoding the {categorical_columns} columns.")

    new_dataframe = pd.DataFrame()

    for column in categorical_columns:
        train_categorical_features = pd.DataFrame(
            encoders[column].transform(dataframe[column].values.reshape(-1, 1)),
            columns=encoders[column].get_feature_names_out(),
        )
        train_categorical_features = train_categorical_features.add_prefix(column + "_")
        new_dataframe = pd.concat([new_dataframe, train_categorical_features], axis=1)

    new_dataframe = pd.concat(
        [new_dataframe, dataframe.drop(columns=categorical_columns)], axis=1
    )
    return new_dataframe


def _drop_features(dataframe: pd.DataFrame, features: List) -> pd.DataFrame:
    """Excludes features from the given dataframe.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe without the given columns.
    """
    return dataframe.drop(columns=features).reset_index(drop=True)


def load_dataset(path: pathlib.Path, from_aws: bool) -> pd.DataFrame:
    """Loads a dataset from a specific path.

    Args:
        path (pathlib.Path): the path where the dataset is located.
        from_aws (bool): whether the dataset is located in an AWS S3 bucket.

    Returns:
        pd.DataFrame: the dataframe.
    """
    logger.info(f"Loading dataset from path {path}.")

    if not from_aws:
        return pd.read_csv(path, sep=",")

    # configuring AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = aws_credentials.AWS_ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_credentials.AWS_SECRET_KEY

    # downloading dataset
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.AWS_ACCESS_KEY,
        aws_secret_access_key=aws_credentials.AWS_SECRET_KEY,
    )

    upload_path = pathlib.Path.joinpath(general_settings.DATA_PATH, path.split("/")[-1])

    s3.download_file(path, upload_path)

    return load_dataset(path=upload_path, from_aws=False)
