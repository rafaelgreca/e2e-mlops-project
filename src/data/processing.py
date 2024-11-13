import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .utils import load_feature
from ..config.settings import general_settings
from ..config.model import model_settings


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

    # Creating the BMR feature
    logger.info("Creating a new column for the BMR values from the data samples.")
    dataframe = _create_bmr_feature(dataframe)

    # Creating the IS feature
    logger.info("Creating a new column for the IS values from the data samples.")
    dataframe = _create_is_feature(dataframe)

    # Feature transformation step)
    # Dropping unused columns
    columns_to_drop = [
        "Height",
        "Weight",
    ]
    logger.info(f"Dropping the columns {columns_to_drop}.")
    dataframe = _drop_features(dataframe=dataframe, features=columns_to_drop)

    # Transforming the AGE and IS columns into a categorical columns
    logger.info("Categorizing the numerical columns ('Age' and 'IS').")
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
    sc = load_feature(path=general_settings.ARTIFACTS_PATH, feature_name="features_sc")

    # Scaling numerical columns
    dataframe = _scale_numerical_columns(dataframe=dataframe, sc=sc)

    # Encoding categorical columns
    dataframe = _encode_categorical_columns(
        dataframe=dataframe,
        encoders=encoders,
        target_column=general_settings.TARGET_COLUMN,
    )

    # Selecting only the features that are important for the model
    dataframe = dataframe[model_settings.FEATURES]
    logger.info(
        f"Filtering the features columns, keeping only {model_settings.FEATURES} columns."
    )

    X = dataframe.values
    return X


def _drop_features(dataframe: pd.DataFrame, features: List) -> pd.DataFrame:
    """Excludes features from the given dataframe.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe without the given columns.
    """
    return dataframe.drop(columns=features).reset_index(drop=True)


def _remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicates.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe without duplicated rows.
    """
    return dataframe.drop_duplicates(keep="first").reset_index(drop=True)


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


def _remove_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes outliers based on the age.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe without outliers.
    """
    # Calculating the upper and lower limits
    q1 = dataframe["Age"].quantile(0.25)
    q3 = dataframe["Age"].quantile(0.75)
    threshold = 3.5
    iqr = q3 - q1

    # Removing the data samples that exceeds the upper or lower limits
    dataframe = dataframe[
        ~(
            (dataframe["Age"] >= (q3 + threshold * iqr))
            | (dataframe["Age"] <= (q1 - threshold * iqr))
        )
    ]

    return dataframe


def _create_is_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Is Sedentary? (IS) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of IS for each data.
    """
    dataframe["IS"] = dataframe["FAF"] <= 1
    dataframe["IS"] = dataframe["IS"].astype(int)
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


def _create_bmr_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Basal Metabolic Rate (BMR) feature.

    Args:
        dataframe (pd.DataFrame): the dataframe.

    Returns:
        pd.DataFrame: the dataframe with a new column corresponding to the
            value of BMR for each data.
    """

    def _calculate_bmr(age: int, gender: str, height: float, weight: float) -> float:
        """Auxiliary function used to calculate the BMR value.

        Args:
            age (int): the person's age.
            gender (str): the person's gender.
            height (float): the person's height.
            weight (float): the person's weight.

        Returns:
            float: the BMR value.
        """
        s = -161 if gender == "Female" else 5
        return (10 * weight) + (6.25 * height) - (5 * age) + s

    dataframe["BMR"] = dataframe.apply(
        lambda x: _calculate_bmr(x["Age"], x["Gender"], x["Height"], x["Weight"]),
        axis=1,
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
    dataframe["IS"] = dataframe["IS"].astype("object")
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
        pd.DataFrame: the dataframe with all numerical columns transformed.
    """
    numerical_columns = dataframe.select_dtypes(exclude="object").columns.tolist()
    logger.info(f"Applying Log Transformation to the {numerical_columns} columns.")

    for nc in numerical_columns:
        dataframe[nc] = np.log1p(dataframe[nc].values + epsilon)

    return dataframe


def _scale_numerical_columns(
    dataframe: pd.DataFrame,
    sc: Dict[str, StandardScaler],
) -> pd.DataFrame:
    """Scales the numerical columns using the Standard technique.

    Args:
        dataframe (pd.DataFrame): the dataframe.
        sc (Dict[str, OneHotEncoder]): a dict containing the corresponding
            encoder for each feature.

    Returns:
        pd.DataFrame: the dataframe with all numerical columns encoded.
    """
    numerical_columns = dataframe.select_dtypes(exclude="object").columns.tolist()
    logger.info(f"Scaling the {numerical_columns} columns.")

    for nc in numerical_columns:
        dataframe[nc] = sc[nc].transform(dataframe[nc].values.reshape(-1, 1))

    return dataframe


def _encode_categorical_columns(
    dataframe: pd.DataFrame,
    encoders: Dict[str, OneHotEncoder],
    target_column: str,
) -> pd.DataFrame:
    """Encodes the categorical columns using the OneHot technique.

    Args:
        dataframe (pd.DataFrame): the dataframe.
        encoders (Dict[str, OneHotEncoder]): a dict containing the corresponding
            encoder for each feature.
        target_column (str): what column is the target label.

    Returns:
        pd.DataFrame: the dataframe with all categorical columns encoded.
    """
    categorical_columns = dataframe.select_dtypes(include="object").columns.tolist()
    # categorical_columns.remove(target_column)
    logger.info(f"Encoding the {categorical_columns} columns.")

    new_dataframe = pd.DataFrame()

    for cc in categorical_columns:
        train_categorical_features = pd.DataFrame(
            encoders[cc].transform(dataframe[cc].values.reshape(-1, 1)),
            columns=encoders[cc].get_feature_names_out(),
        )
        train_categorical_features = train_categorical_features.add_prefix(cc + "_")
        new_dataframe = pd.concat([new_dataframe, train_categorical_features], axis=1)

    new_dataframe = pd.concat(
        [new_dataframe, dataframe.drop(columns=categorical_columns)], axis=1
    )
    return new_dataframe


def _encode_labels_array(
    array: np.ndarray,
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """Encodes an array containing the labels (e.g., transform strings to OneHotEncoder).

    Args:
        array (np.ndarray): the labels array.
        encoder (OneHotEncoder): the OneHotEncoder instance.

    Returns:
        pd.DataFrame: the encoded array.
    """
    return encoder.transform(array.reshape(-1, 1))


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Loads a dataset from a specific path.

    Args:
        path (pathlib.Path): the path where the dataset is located.

    Returns:
        pd.DataFrame: the dataframe.
    """
    logger.info(f"Loading dataset from path {path}.")
    return pd.read_csv(path, sep=",")
