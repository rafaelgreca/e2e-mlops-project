import yaml
from pathlib import Path
from typing import Dict, Optional, Type, Any, Tuple
from copy import deepcopy

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo


def partial_model(model: Type[BaseModel]):
    """Workaround for setting all Pydantic's fields as optional.
    All credits goes to the author:
    https://stackoverflow.com/questions/67699451/make-every-field-as-optional-with-pydantic

    Args:
        model (Type[BaseModel]): Pydantic base model instance.
    """

    def make_field_optional(
        field: FieldInfo, default: Any = None
    ) -> Tuple[Any, FieldInfo]:
        new = deepcopy(field)
        new.default = default
        new.annotation = Optional[field.annotation]  # type: ignore
        return new.annotation, new

    return create_model(
        f"Partial{model.__name__}",
        __base__=model,
        __module__=model.__module__,
        **{
            field_name: make_field_optional(field_info)
            for field_name, field_info in model.model_fields.items()
        },
    )


def read_yaml_credentials_file(file_path: Path, file_name: str) -> Dict:
    """Reads a YAML file.

    Args:
        file_path (Path): the file's path.
        file_name (str): the file's name.

    Raises:
        e: If any error occurs when trying to read the YAML
            file, then returns the error to the user.

    Returns:
        Dict: the content of the YAML file.
    """
    path = Path.joinpath(
        file_path,
        file_name,
    )

    with open(path, "r", encoding="utf-8") as f:
        try:
            context = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise e

    return context
