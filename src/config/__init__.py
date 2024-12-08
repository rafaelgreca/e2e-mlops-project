"""
Stores auxiliary functions (such as for reading an YAML file)
that will be used with the main configurations functions.
"""
from pathlib import Path
from typing import Dict

import yaml


def read_yaml_credentials_file(file_path: Path, file_name: str) -> Dict:
    """Reads a YAML file.

    Args:
        file_path (Path): the file's path.
        file_name (str): the file's name.

    Raises:
        error: If any error occurs when trying to read the YAML
            file, then returns the error to the user.

    Returns:
        Dict: the content of the YAML file.
    """
    path = Path.joinpath(
        file_path,
        file_name,
    )

    with open(path, "r", encoding="utf-8") as file:
        try:
            context = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise error

    return context
