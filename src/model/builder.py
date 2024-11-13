import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..config.model import model_settings
from ..data.processing import data_processing


class ModelBuilder:
    """The trained model's class."""

    def __init__(
        self,
        model_name: str,
        model_flavor: str,
        model_version: str,
    ) -> None:
        """Model's instance initializer.

        Args:
            model_name (str): the model's name.
            model_flavor (str): the model's MLflow flavor.
            model_version (str): the model's version.
        """
        self.model_name = model_name
        self.model_flavor = model_flavor
        self.model_version = model_version
        self.model = None

    @logger.catch
    def train(self, dataframe: pd.DataFrame) -> None:
        logger.info("Pre-processing the data before training the model.")

        # Pre-processing and cleaning the data
        X, y = data_processing(dataframe)

        logger.info(
            "Splitting the data into training and validation using 90/10 split."
        )

        # Splitting the data into training and validation
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.1,
            shuffle=True,
            stratify=y,
        )

        logger.info("Training the model using the given data.")
        self.model = XGBClassifier()
        self.model.fit(X_train, y_train)

        # Assessing the model's performance on the training set
        train_prediction = np.argmax(self.model.predict(X_train), axis=1)
        _y_train = np.argmax(y_train, axis=1).reshape(-1)
        score = f1_score(y_true=_y_train, y_pred=train_prediction, average="weighted")
        logger.info(f"Achieved a weighted F1-Score of {score} on the training set.")

        # Assessing the model's performance on the validation set
        valid_prediction = np.argmax(self.model.predict(X_valid), axis=1)
        _y_valid = np.argmax(y_valid, axis=1).reshape(-1)
        score = f1_score(y_true=_y_valid, y_pred=valid_prediction, average="weighted")
        logger.info(f"Achieved a weighted F1-Score of {score} on the validation set.")
