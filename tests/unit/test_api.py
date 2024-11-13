import json
from pathlib import Path
from typing import Dict

import requests

from src.config.model import model_settings
from src.config.settings import general_settings

with open(
    f"{Path.joinpath(general_settings.RESEARCH_ENVIRONMENT_PATH, 'VERSION')}",
    "r",
    encoding="utf-8",
) as f:
    CODE_VERSION = f.readline().strip()


def test_version_endpoint() -> None:
    """
    Unit case to test the API's version endpoint.
    """
    desired_keys = ["model_version", "code_version"]

    response = requests.get("http://127.0.0.1:8000/version", timeout=100)
    content = json.loads(response.text)

    assert response.status_code == 200
    assert isinstance(content, Dict)
    assert all(dk in content.keys() for dk in desired_keys)
    assert model_settings.VERSION == content[desired_keys[0]]
    assert CODE_VERSION == content[desired_keys[1]]


def test_inference_endpoint() -> None:
    """
    Unit case to test the API's inference endpoint.
    """
    desired_classes = [["Normal_Weight"]]
    desired_keys = ["predictions"]

    data = {
        "Age": 21,
        "CAEC": "Sometimes",
        "CALC": "no",
        "FAF": 0,
        "FCVC": 2,
        "Gender": "Female",
        "Height": 1.62,
        "MTRANS": "Public_Transportation",
        "SCC": "no",
        "SMOKE": "False",
        "TUE": 1,
        "Weight": 64,
    }

    response = requests.get("http://127.0.0.1:8000/predict", json=data, timeout=100)
    content = json.loads(response.text)

    assert response.status_code == 200
    assert isinstance(content, Dict)
    assert all(dk in content.keys() for dk in desired_keys)
    assert content[desired_keys[0]] == desired_classes
