"""
Stores auxiliary functions (such as for loading features or downloading the dataset)
that will be used with the main data processing functions.
"""
import pathlib
import os
from typing import Union

import boto3
import joblib
import numpy as np
from loguru import logger
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


@logger.catch
def download_dataset(
    name: str,
    new_name: str,
    path: pathlib.Path,
    send_to_aws: bool,
    file_type: str,
) -> None:
    """Dowload the dataset using Kaggle's API.

    Args:
        name (str): the dataset's name.
        new_name (str): the dataset file's new name.
        path (pathlib.Path): the path where the dataset will be stored locally.
        send_to_aws (bool): whether the dataset will be send to an AWS S3 bucket or not.
        file_type (str): what kind of dataset will be downloaded ('raw' or 'current').
    """
    os.environ["KAGGLE_USERNAME"] = kaggle_credentials.KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = kaggle_credentials.KAGGLE_KEY

    logger.info(f"Downloading dataset {name} and saving into the folder {path}.")

    # Downloading data using the Kaggle API through the terminal
    if file_type == "current":
        os.system(f"kaggle datasets download -d {name} --unzip")
        os.system(f"mv ObesityDataSet.csv {pathlib.Path.joinpath(path, new_name)}")
    elif file_type == "raw":
        os.system(f"kaggle competitions download -c {name}")
        os.system(f"unzip {name}.zip")

        # deleting the zip file, the sample submission file, and
        # the test file, as we are only using the training data for now
        os.system(f"rm {name}.zip sample_submission.csv test.csv")

        os.system(f"mv train.csv {pathlib.Path.joinpath(path, new_name)}")
    else:
        raise ValueError("The value for 'file_type' must be 'raw' or 'current'.\n")

    # Sending the dataset to the AWS S3 bucket
    if send_to_aws:
        if aws_credentials.S3 != "YOUR_S3_BUCKET_URL":
            send_dataset_to_s3(
                file_path=path,
                file_name=new_name,
            )
        else:
            logger.warning(
                "The S3 Bucket url was not specified in the 'credentials.yaml' file. "
                + "Therefore, the dataset will not be send to S3 and it will be kept saved locally."
            )


@logger.catch
def send_dataset_to_s3(
    file_path: pathlib.Path,
    file_name: str,
) -> None:
    """Sends a given dataset to the AWS S3 Bucket.

    Args:
        file_path (pathlib.Path): the dataset file's path.
        file_name (str): the file's name.
    """
    bucket = boto3.client(
        "s3",
        aws_access_key_id=aws_credentials.AWS_ACCESS_KEY,
        aws_secret_access_key=aws_credentials.AWS_SECRET_KEY,
    )

    bucket.upload_file(
        file_path,
        aws_credentials.S3,
        file_name,
    )

    os.remove(pathlib.Path.joinpath(file_path, file_name))
