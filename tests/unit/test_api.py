"""
Unit test cases to test the API code.
"""
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


def test_model_performance_report_endpoint() -> None:
    """
    Unit case to test the API's model performance report endpoint.
    """
    window_size = 300
    path = "results/model_performance.html"
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://127.0.0.1:8000/monitor-model?window_size={window_size}",
        timeout=100,
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert Path.exists(Path(path))


def test_target_drift_report_endpoint() -> None:
    """
    Unit case to test the API's target drift report endpoint.
    """
    window_size = 300
    path = "results/target_drift.html"
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://127.0.0.1:8000/monitor-target?window_size={window_size}",
        timeout=100,
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert Path.exists(Path(path))


def test_data_drift_report_endpoint() -> None:
    """
    Unit case to test the API's data drift report endpoint.
    """
    window_size = 300
    path = "results/data_drift.html"
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://127.0.0.1:8000/monitor-data?window_size={window_size}",
        timeout=100,
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert Path.exists(Path(path))


def test_data_quality_report_endpoint() -> None:
    """
    Unit case to test the API's data quality report endpoint.
    """
    window_size = 300
    path = "results/data_quality.html"
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://127.0.0.1:8000/monitor-data-quality?window_size={window_size}",
        timeout=100,
        headers=headers,
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert Path.exists(Path(path))


def test_inference_endpoint() -> None:
    """
    Unit case to test the API's inference endpoint.
    """
    desired_classes = ["Normal_Weight"]
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

    response = requests.post("http://127.0.0.1:8000/predict", json=data, timeout=100)
    content = json.loads(response.text)

    assert response.status_code == 200
    assert isinstance(content, Dict)
    assert all(dk in content.keys() for dk in desired_keys)
    assert content[desired_keys[0]] == desired_classes
