import os
import pathlib
import joblib
from typing import Union

import boto3
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ..config.aws import aws_credentials
from ..config.kaggle import kaggle_credentials


def load_feature(
    path: pathlib.Path,
    feature_name: str,
) -> Union[np.ndarray, StandardScaler, OneHotEncoder]:
    """Loads a given feature (might be a Numpy's array or a Scikit-learn's encoder/scaler).

    Args:
        path (pathlib.Path): the path of the desired feature.
        feature_name (str): the feature file's name.

    Returns:
        Union[np.ndarray, StandardScaler, OneHotEncoder]: the feature's content.
    """
    logger.info(f"Loading feature/encoder/scaler from file {path}.")
    return joblib.load(pathlib.PosixPath.joinpath(path, f"{feature_name}.pkl"))


def custom_combiner(feature, category) -> str:
    """Auxiliary function that is used to rename the output's columns from
    the OneHotEncoder instance.

    Args:
        feature (_type_): the feature's name (ignored).
        category (_type_): the category's name.

    Returns:
        str: the current category from that given feature.
    """
    return str(category)

@logger.catch
def download_dataset(
    name: str,
) -> None:
    """Dowload the dataset using Kaggle's API.

    Args:
        name (str): the dataset's name.
    """
    kaggle_user = kaggle_credentials.KAGGLE_USERNAME
    kaggle_key = kaggle_credentials.KAGGLE_KEY
    path = '../data/'
    logger.info(f"Downloading dataset {name} and saving into the folder {path}.")

    # Downloading data using the Kaggle API through the terminal
    os.system(f'export KAGGLE_USERNAME={kaggle_user}; export KAGGLE_KEY={kaggle_key};')
    os.system(f'kaggle datasets download -d {name} -p {path} --unzip')

    # Sending the dataset to the AWS S3 bucket
    if aws_credentials.S3 != "YOUR_S3_BUCKET_URL":
        send_dataset_to_s3()


def send_dataset_to_s3(
    file_path: pathlib.Path,
    file_name: str,
) -> None:
    """Sends a given dataset to the AWS S3 Bucket.

    Args:
        file_path (pathlib.Path): the dataset file's path.
        file_name (str): the file's name.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.AWS_ACCESS_KEY,
        aws_secret_access_key=aws_credentials.AWS_SECRET_KEY,
    )

    s3.upload_file(
        file_path,
        aws_credentials.S3,
        file_name,
    )
