"""
API's main file.
"""
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from loguru import logger

from .utils import (
    build_data_drift_report,
    build_data_quality_report,
    build_model_performance_report,
    build_target_drift_report,
    get_column_mapping,
)
from ..data.processing import data_processing_inference
from ..config.model import model_settings
from ..config.reports import report_settings
from ..config.settings import general_settings
from ..schema.person import Person
from ..schema.monitoring import Monitoring
from . import current_dataset, loaded_model, reference_data

app = FastAPI()


@app.get("/monitor-model")
def monitor_model_performance(monitoring: Monitoring = Depends()) -> FileResponse:
    """
    This endpoint is used to create a report for monitoring model performance.

    Returns:
        FileResponse: the report HTML file.
    """
    window_size = monitoring.window_size

    logger.info(f"Loading current data and selecting the first {window_size} rows.")
    current_data = current_dataset.head(window_size).copy()
    current_data = current_data.drop(columns=[general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=current_data)
    current_data = pd.DataFrame(features, columns=model_settings.FEATURES)
    current_data[general_settings.TARGET_COLUMN] = current_dataset[
        general_settings.TARGET_COLUMN
    ].copy()

    current_data["prediction"] = loaded_model.predict(features)

    column_mapping = get_column_mapping(
        dataframe=current_data,
        target_column=general_settings.TARGET_COLUMN,
        features=model_settings.FEATURES,
        predict_column="prediction",
    )

    logger.info("Building the model performance report.")
    report_path = build_model_performance_report(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=column_mapping,
        report_path=Path.joinpath(
            report_settings.REPORTS_PATH, report_settings.MODEL_PERFORMANCE_REPORT_NAME
        ),
    )

    logger.info(f"Returning report as HTML file in location {report_path}.")
    return FileResponse(report_path)


@app.get("/monitor-target")
def monitor_target_drift(monitoring: Monitoring = Depends()) -> FileResponse:
    """
    This endpoint is used to create a report for monitoring target drift.

    Returns:
        FileResponse: the report HTML file.
    """
    window_size = monitoring.window_size

    logger.info(f"Loading current data and selecting the first {window_size} rows.")
    current_data = current_dataset.head(window_size).copy()
    current_data = current_data.drop(columns=[general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=current_data)
    current_data = pd.DataFrame(features, columns=model_settings.FEATURES)
    current_data[general_settings.TARGET_COLUMN] = current_dataset[
        general_settings.TARGET_COLUMN
    ].copy()

    current_data["prediction"] = loaded_model.predict(features)

    column_mapping = get_column_mapping(
        dataframe=current_data,
        target_column=general_settings.TARGET_COLUMN,
        features=model_settings.FEATURES,
        predict_column="prediction",
    )

    logger.info("Building the target drift report.")
    report_path = build_target_drift_report(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=column_mapping,
        report_path=Path.joinpath(
            report_settings.REPORTS_PATH, report_settings.TARGET_DRIFT_REPORT_NAME
        ),
    )

    logger.info(f"Returning report as HTML file in location {report_path}.")
    return FileResponse(report_path)


@app.get("/monitor-data")
def monitor_data_drift(monitoring: Monitoring = Depends()) -> FileResponse:
    """
    This endpoint is used to create a report for monitoring data drift.

    Returns:
        FileResponse: the report HTML file.
    """
    window_size = monitoring.window_size

    logger.info(f"Loading current data and selecting the first {window_size} rows.")
    current_data = current_dataset.head(window_size).copy()
    current_data = current_data.drop(columns=[general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=current_data)
    current_data = pd.DataFrame(features, columns=model_settings.FEATURES)
    current_data[general_settings.TARGET_COLUMN] = current_dataset[
        general_settings.TARGET_COLUMN
    ].copy()

    current_data["prediction"] = loaded_model.predict(features)

    column_mapping = get_column_mapping(
        dataframe=current_data,
        target_column=general_settings.TARGET_COLUMN,
        features=model_settings.FEATURES,
        predict_column="prediction",
    )

    logger.info("Building the data drift report.")
    report_path = build_data_drift_report(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=column_mapping,
        report_path=Path.joinpath(
            report_settings.REPORTS_PATH, report_settings.DATA_DRIFT_REPORT_NAME
        ),
    )

    logger.info(f"Returning report as HTML file in location {report_path}.")
    return FileResponse(report_path)


@app.get("/monitor-data-quality")
def monitor_data_quality(monitoring: Monitoring = Depends()) -> FileResponse:
    """
    This endpoint is used to create a report for monitoring data quality.

    Returns:
        FileResponse: the report HTML file.
    """
    window_size = monitoring.window_size

    logger.info(f"Loading current data and selecting the first {window_size} rows.")
    current_data = current_dataset.head(window_size).copy()
    current_data = current_data.drop(columns=[general_settings.TARGET_COLUMN])

    features = data_processing_inference(dataframe=current_data)
    current_data = pd.DataFrame(features, columns=model_settings.FEATURES)
    current_data[general_settings.TARGET_COLUMN] = current_dataset[
        general_settings.TARGET_COLUMN
    ].copy()

    current_data["prediction"] = loaded_model.predict(features)

    column_mapping = get_column_mapping(
        dataframe=current_data,
        target_column=general_settings.TARGET_COLUMN,
        features=model_settings.FEATURES,
        predict_column="prediction",
    )

    logger.info("Building the data quality report.")
    report_path = build_data_quality_report(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=column_mapping,
        report_path=Path.joinpath(
            report_settings.REPORTS_PATH, report_settings.DATA_QUALITY_REPORT_NAME
        ),
    )

    logger.info(f"Returning report as HTML file in location {report_path}.")
    return FileResponse(report_path)


@app.get("/version")
def check_versions() -> Dict:
    """
    This endpoint will return the current model and code versions.

    Returns:
        Dict: the model and code versions.
    """
    with open(
        f"{general_settings.RESEARCH_ENVIRONMENT_PATH}/VERSION", "r", encoding="utf-8"
    ) as file:
        code_version = file.readline().strip()

    return {
        "code_version": code_version,
        "model_version": model_settings.VERSION,
    }


@app.post("/predict")
async def prediction(person: Person) -> Dict:
    """
    This endpoint is used to make a prediction (with the trained model)
    with the given data.

    Args:
        person (Person): a person's data.

    Returns:
        Dict: the predictions.
    """
    data = pd.DataFrame.from_dict([person.model_dump()])
    features = data_processing_inference(data)

    return {"predictions": loaded_model.predict(features).tolist()}
