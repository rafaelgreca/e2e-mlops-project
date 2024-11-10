import pathlib

from .data.processing import load_dataset, data_processing
from .data.utils import custom_combiner
from .config.settings import general_settings
from .config.model import model_settings
from .model import Model

if __name__ == "__main__":
    dataset = load_dataset(
        path=pathlib.Path.joinpath(general_settings.DATA_PATH, general_settings.RAW_FILE_NAME)
    )
    X, y = data_processing(dataset)

    loaded_model = Model(
        model_name=model_settings.MODEL_NAME,
        model_flavor=model_settings.MODEL_FLAVOR,
        model_version=model_settings.VERSION,
    )
    loaded_model.load()


    print(loaded_model.predict(X))
    print(loaded_model.score(X, y))