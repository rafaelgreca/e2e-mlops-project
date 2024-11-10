import mlflow
import numpy as np
import pathlib
from sklearn.metrics import f1_score

from .config.model import model_settings
from .data.processing import load_dataset, data_processing
from .data.utils import custom_combiner
from .config.settings import general_settings

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

    def load(self) -> None:
        """Loads the trained model.

        Raises:
            NotImplementedError: raises NotImplementedError if the model's
                flavor value is not 'xgboost'.
        """
        if self.model_flavor == "xgboost":
            model_uri = f"runs:/{model_settings.RUN_ID}/{model_settings.MODEL_NAME}"
            self.model = mlflow.xgboost.load_model(model_uri)
        else:
            raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Uses the trained model to make a prediction on a given feature array.

        Args:
            x (np.ndarray): the features array.

        Returns:
            np.ndarray: the predictions array.
        """
        return np.argmax(self.model.predict(x), axis=1)

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

        return f1_score(y_true=_y, y_pred=prediction, average="weighted")

# if __name__ == "__main__":
#     dataset = load_dataset(
#         path=pathlib.Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME)
#     )
#     X, y = data_processing(dataset)

#     loaded_model = Model(
#         model_name=model_settings.MODEL_NAME,
#         model_flavor=model_settings.MODEL_FLAVOR,
#         model_version=model_settings.VERSION,
#     )
#     loaded_model.load()


#     print(loaded_model.predict(X))
#     print(loaded_model.score(X, y))