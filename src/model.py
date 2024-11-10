import mlflow
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score

from .config.model import model_settings


class Model:
    """The trained model's class.
    """
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
    def load(self) -> None:
        """Loads the trained model.

        Raises:
            NotImplementedError: raises NotImplementedError if the model's
                flavor value is not 'xgboost'.
        """
        logger.info(f"Loading the model {model_settings.MODEL_NAME} from run ID {model_settings.RUN_ID}.")

        if self.model_flavor == "xgboost":
            model_uri = f"runs:/{model_settings.RUN_ID}/{model_settings.MODEL_NAME}"
            self.model = mlflow.xgboost.load_model(model_uri)
        else:
            logger.critical(f"Couldn't load the model using the flavor {model_settings.MODEL_FLAVOR}.")
            raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Uses the trained model to make a prediction on a given feature array.

        Args:
            x (np.ndarray): the features array.

        Returns:
            np.ndarray: the predictions array.
        """
        prediction = np.argmax(self.model.predict(x), axis=1)
        logger.info(f"Prediction: {prediction}.")
        return prediction

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculates the F1-Score of a trained model given a pair of features
        and labels arrays.

        Args:
            x (np.ndarray): the features array.
            y (np.ndarray): the targets array.

        Returns:
            float: the F1 Score value.
        """
        prediction = self.predict(x).reshape(-1)
        _y = np.argmax(y, axis=1).reshape(-1)

        score = f1_score(y_true=_y, y_pred=prediction, average="weighted")
        logger.info(f"Achieved a weighted F1-Score of {score}.")
        return score
