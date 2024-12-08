"""
Unit test cases to test the API code.
"""
import json
from pathlib import Path
from typing import Dict

import requests

from src.config.model import model_settings
from src.config.reports import report_settings
from . import CODE_VERSION


def test_version_endpoint() -> None:
    """
    Unit case to test the API's version endpoint.
    """
    desired_keys = ["model_version", "code_version"]

    response = requests.get("http://prod:8000/version", timeout=100)
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
    path = Path.joinpath(
        report_settings.REPORTS_PATH, report_settings.MODEL_PERFORMANCE_REPORT_NAME
    )
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://prod:8000/monitor-model?window_size={window_size}",
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
    path = Path.joinpath(
        report_settings.REPORTS_PATH, report_settings.TARGET_DRIFT_REPORT_NAME
    )
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://prod:8000/monitor-target?window_size={window_size}",
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
    path = Path.joinpath(
        report_settings.REPORTS_PATH, report_settings.DATA_DRIFT_REPORT_NAME
    )
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://prod:8000/monitor-data?window_size={window_size}",
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
    path = Path.joinpath(
        report_settings.REPORTS_PATH, report_settings.DATA_QUALITY_REPORT_NAME
    )
    headers = {"Accept-Encoding": "identity"}

    response = requests.get(
        f"http://prod:8000/monitor-data-quality?window_size={window_size}",
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
    desired_classes = ["Overweight_Level_II"]
    desired_keys = ["predictions"]

    data = {
        "Age": 24.443011,
        "Height": 1.699998,
        "Weight": 81.66995,
        "Gender": "Male",
        "family_history_with_overweight": "yes",
        "CALC": "Sometimes",
        "MTRANS": "Public_Transportation",
        "FAVC": "yes",
        "FCVC": 2,
        "NCP": 2.983297,
        "CH2O": 2.763573,
        "FAF": 0,
        "TUE": 1,
        "CAEC": "Sometimes",
        "SCC": "no",
    }

    response = requests.post("http://prod:8000/predict", json=data, timeout=100)
    content = json.loads(response.text)

    assert response.status_code == 200
    assert isinstance(content, Dict)
    assert all(dk in content.keys() for dk in desired_keys)
    assert content[desired_keys[0]] == desired_classes
