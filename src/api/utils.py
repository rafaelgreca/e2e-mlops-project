"""
Auxiliary functions used to generate monitoring reports.
"""
from typing import List, Text
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
from evidently import ColumnMapping
from evidently.metrics import (
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    ClassificationQualityMetric,
    DatasetCorrelationsMetric,
    DatasetMissingValuesMetric,
    DatasetSummaryMetric,
)
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report


def get_column_mapping(
    dataframe: pd.DataFrame,
    target_column: str,
    features: List[str],
    predict_column: str,
) -> ColumnMapping:
    """
    Generates a column mapping object that will be used to create
    model, data, and target monitoring reports.

    Args:
        dataframe (pd.DataFrame): the current dataframe.
        target_column (str): the target column's name.
        features (List[str]): a list containing all the features that
            are being used.
        predict_column (str): the prediction column's name.

    Returns:
        ColumnMapping: the created column mapping object.
    """
    numerical_columns = []
    categorical_columns = []

    for feature in features:
        if ptypes.is_numeric_dtype(dataframe[feature]):
            numerical_columns.append(feature)
        else:
            categorical_columns.append(feature)

    column_mapping = ColumnMapping()
    column_mapping.target = target_column
    column_mapping.prediction = predict_column
    column_mapping.categorical_features = categorical_columns
    column_mapping.numerical_features = numerical_columns

    return column_mapping


def build_model_performance_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    report_path: Path,
) -> Text:
    """
    Builds a Model Performance Report.

    Args:
        current_data (pd.DataFrame): the current data.
        reference_data (pd.DataFrame): the reference data (the data used
            to train the model).
        column_mapping (ColumnMapping): the column mapping.
        report_path (Path): where the reported will be saved.

    Returns:
        Text: the reported path.
    """
    model_performance_report = Report(
        metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass(),
        ]
    )
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    model_performance_report.save_html(str(report_path))
    return report_path


def build_target_drift_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    report_path: Path,
) -> Text:
    """
    Builds a Target Drift Report.

    Args:
        current_data (pd.DataFrame): the current data.
        reference_data (pd.DataFrame): the reference data (the data used
            to train the model).
        column_mapping (ColumnMapping): the column mapping.
        report_path (Path): where the reported will be saved.

    Returns:
        Text: the reported path.
    """
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    target_drift_report.save_html(str(report_path))
    return report_path


def build_data_drift_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    report_path: Path,
) -> Text:
    """
    Builds a Data Drift Report.

    Args:
        current_data (pd.DataFrame): the current data.
        reference_data (pd.DataFrame): the reference data (the data used
            to train the model).
        column_mapping (ColumnMapping): the column mapping.
        report_path (Path): where the reported will be saved.

    Returns:
        Text: the reported path.
    """
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    data_drift_report.save_html(str(report_path))
    return report_path


def build_data_quality_report(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    report_path: Path,
) -> Text:
    """
    Builds a Data Quality Report.

    Args:
        current_data (pd.DataFrame): the current data.
        reference_data (pd.DataFrame): the reference data (the data used
            to train the model).
        column_mapping (ColumnMapping): the column mapping.
        report_path (Path): where the reported will be saved.

    Returns:
        Text: the reported path.
    """
    data_quality_report = Report(
        metrics=[
            DataQualityPreset(),
            DatasetMissingValuesMetric(),
            DatasetCorrelationsMetric(),
            DatasetSummaryMetric(),
        ]
    )
    data_quality_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    data_quality_report.save_html(str(report_path))
    return report_path
