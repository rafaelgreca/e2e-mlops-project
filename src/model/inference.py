import mlflow
import numpy as np
from loguru import logger

from ..config.model import model_settings
from ..config.settings import general_settings
from ..data.utils import load_feature

label_encoder = load_feature(
    path=general_settings.ARTIFACTS_PATH,
    feature_name='label_ohe'
)

class ModelServe:
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
        prediction = self.model.predict(x)
        prediction = label_encoder.inverse_transform(prediction)
        logger.info(f"Prediction: {prediction}.")
        return prediction
