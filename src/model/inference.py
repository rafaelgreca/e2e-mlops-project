"""
Stores a model serve class that will be used to make predictions with
the trained model.
"""
import mlflow
import numpy as np
from loguru import logger

from ..config.aws import aws_credentials
from ..config.model import model_settings
from ..config.settings import general_settings
from ..data.utils import load_feature

label_encoder = load_feature(
    path=general_settings.ARTIFACTS_PATH, feature_name="label_ohe"
)

if aws_credentials.EC2 != "YOUR_EC2_INSTANCE_URL":
    mlflow.set_tracking_uri(f"http://{aws_credentials.EC2}:5000")
else:
    mlflow.set_tracking_uri("http://mlflow:5000")


class ModelServe:
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
    def load(self) -> None:
        """Loads the trained model.

        Raises:
            NotImplementedError: raises NotImplementedError if the model's
                flavor value is not 'lightbm'.
        """
        logger.info(
            f"Loading the model {model_settings.MODEL_NAME} from run ID {model_settings.RUN_ID}."
        )

        if self.model_flavor == "lightgbm":
            model_uri = f"runs:/{model_settings.RUN_ID}/{model_settings.MODEL_NAME}"
            self.model = mlflow.lightgbm.load_model(model_uri)
        else:
            logger.critical(
                f"Couldn't load the model using the flavor {model_settings.MODEL_FLAVOR}."
            )
            raise NotImplementedError()

    def predict(
        self, features: np.ndarray, transform_to_str: bool = True
    ) -> np.ndarray:
        """Uses the trained model to make a prediction on a given feature array.

        Args:
            features (np.ndarray): the features array.
            transform_to_str (bool): whether to transform the prediction integer to
                string or not. Defaults to True.

        Returns:
            np.ndarray: the predictions array.
        """
        prediction = self.model.predict(features)

        if transform_to_str:
            one_hot = np.zeros((prediction.size, prediction.max() + 1))
            one_hot[np.arange(prediction.size), prediction] = 1
            prediction = label_encoder.inverse_transform(one_hot)

        logger.info(f"Prediction: {prediction}.")
        return prediction
