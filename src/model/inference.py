"""
Stores a model serve class that will be used to make predictions with
the trained model.
"""
import os
import pickle
import pathlib
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
        """Loads the trained model from MLflow or local filesystem as fallback.

        Raises:
            NotImplementedError: raises NotImplementedError if the model's
                flavor value is not supported.
            RuntimeError: if model cannot be loaded from either MLflow or local.
        """
        # Try loading from MLflow first
        try:
            self._load_from_mlflow()
            return
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow: {e}")
            logger.info("Attempting to load model from local filesystem...")
        
        # Fallback to local filesystem
        try:
            self._load_from_local()
            logger.info("Model loaded successfully from local filesystem")
        except Exception as e:
            logger.error(f"Failed to load model from local filesystem: {e}")
            raise RuntimeError(
                "Failed to load model from both MLflow and local filesystem"
            )

    def _load_from_mlflow(self) -> None:
        """Loads the trained model from MLflow server."""
        logger.info(
            f"Loading the model {model_settings.MODEL_NAME} from run ID {model_settings.RUN_ID}."
        )

        if self.model_flavor == "lightgbm":
            model_uri = f"runs:/{model_settings.RUN_ID}/{model_settings.MODEL_NAME}"
            self.model = mlflow.lightgbm.load_model(model_uri)
            logger.info(f"Model loaded successfully from MLflow: {model_uri}")
        else:
            logger.critical(
                f"Couldn't load the model using the flavor {model_settings.MODEL_FLAVOR}."
            )
            raise NotImplementedError(
                f"Model flavor {self.model_flavor} not supported"
            )

    def _load_from_local(self) -> None:
        """Loads the trained model from local mlartifacts directory."""
        # Determine base paths to search
        possible_base_paths = [
            pathlib.Path("../notebooks/mlartifacts"),  # From src/
            pathlib.Path("./notebooks/mlartifacts"),   # From project root
            pathlib.Path("/e2e-mlops-project/notebooks/mlartifacts"),  # Docker path
            pathlib.Path("./mlartifacts"),  # Alternative path
        ]
        
        model_path = None
        for base_path in possible_base_paths:
            potential_path = (
                base_path 
                / model_settings.EXPERIMENT_ID 
                / model_settings.RUN_ID 
                / "artifacts" 
                / self.model_flavor 
                / "model.pkl"
            )
            logger.debug(f"Checking for model at: {potential_path}")
            if potential_path.exists():
                model_path = potential_path
                logger.info(f"Found model at: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"Model not found in any of the expected locations for "
                f"experiment_id={model_settings.EXPERIMENT_ID}, "
                f"run_id={model_settings.RUN_ID}, "
                f"flavor={self.model_flavor}. "
                f"Searched paths: {[str(p / model_settings.EXPERIMENT_ID / model_settings.RUN_ID / 'artifacts' / self.model_flavor / 'model.pkl') for p in possible_base_paths]}"
            )
        
        # Load the model using pickle
        logger.info(f"Loading model from local file: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from: {model_path}")

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
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        prediction = self.model.predict(features)

        if transform_to_str:
            one_hot = np.zeros((prediction.size, prediction.max() + 1))
            one_hot[np.arange(prediction.size), prediction] = 1
            prediction = label_encoder.inverse_transform(one_hot)

        logger.info(f"Prediction: {prediction}.")
        return prediction
