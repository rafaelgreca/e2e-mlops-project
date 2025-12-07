#!/usr/bin/env python3
"""
Standalone script to run ML model experiments (feature selection + hyperparameter tuning).
This script can be run in the background for long-running training jobs.

Usage:
    python run_experiments.py
    
    Or run in background:
    nohup python run_experiments.py > training.log 2>&1 &
"""

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from pprint import pprint
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
import boto3
import joblib
import os
import sys
import math
import mlflow
import optuna
import warnings
import shutil
from datetime import datetime

import logging
import sys

warnings.filterwarnings("ignore")

# Add project root to path if running as script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def setup_logger(log_file="training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Avoid duplicated logs
    if not logger.handlers:
        logger.addHandler(console)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()


def read_yaml_file(path, file):
    """Reading credentials files."""
    with open(f"{os.path.join(path, file)}") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise e
    return content


def get_or_create_experiment(experiment_name: str, tags: dict = None) -> str:
    """
    Get existing experiment or create a new one if it doesn't exist.
    
    Args:
        experiment_name: Name of the experiment
        tags: Optional tags to add if creating new experiment
        
    Returns:
        Experiment ID as string
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags=tags or {}
            )
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
        else:
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            return experiment.experiment_id
    except Exception as e:
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags=tags or {}
            )
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
        except Exception as create_error:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
                return experiment.experiment_id
            else:
                raise create_error


def apply_feature_selection(
    model: Union[DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier],
    number_features: int,
    X_train: np.ndarray,
    y_train: np.array,
    X_valid: np.ndarray,
    y_valid: np.array,
    FEATURES_NAME: List[str]
) -> Dict:
    """Apply feature selection using SequentialFeatureSelector."""
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=number_features,
        cv=3
    )
    sfs.fit(X=X_train, y=y_train)

    selected_features_indexes = np.argwhere(sfs.get_support()).reshape(-1)

    reduced_X_train = sfs.transform(X_train)
    reduced_X_valid = sfs.transform(X_valid)

    model.fit(reduced_X_train, y_train)

    predicted_y_train = model.predict(reduced_X_train)
    train_f1 = f1_score(
        y_true=y_train,
        y_pred=predicted_y_train,
        average="weighted"
    )
    
    predicted_y_valid = model.predict(reduced_X_valid)
    valid_f1 = f1_score(
        y_true=y_valid,
        y_pred=predicted_y_valid,
        average="weighted"
    )

    signature = infer_signature(
        model_input=reduced_X_train,
        model_output=predicted_y_train
    )
    
    selected_features_names = list(map(lambda i: FEATURES_NAME[i], selected_features_indexes.tolist()))

    results = {
        "train_f1": train_f1,
        "valid_f1": valid_f1,
        "features": selected_features_names,
        "model": model,
        "model_signature": signature
    }

    return results


def set_configurations_mlflow(
    model: Union[DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier],
    y_train: np.array,
    y_valid: np.array,
) -> Tuple[np.array, np.array, str, str]:
    """Configure MLflow settings based on model type."""
    if isinstance(model, DecisionTreeClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "decision_tree"
        flavor = "sklearn"
    
    if isinstance(model, RandomForestClassifier):
        run_name = "random_forest"
        flavor = "sklearn"
    
    if isinstance(model, XGBClassifier):
        run_name = "xgboost"
        flavor = "xgboost"
    
    if isinstance(model, LGBMClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "lightgbm"
        flavor = "lightgbm"
    
    if isinstance(model, CatBoostClassifier):
        y_train = np.argmax(y_train, axis=1)
        y_valid = np.argmax(y_valid, axis=1)
        run_name = "catboost"
        flavor = "catboost"
    
    # Disable autolog
    if flavor == "sklearn":
        mlflow.sklearn.autolog(
            log_models=False,
            log_post_training_metrics=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True
        )
    elif flavor == "xgboost":
        mlflow.xgboost.autolog(
            log_models=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True
        )
    elif flavor == "lightgbm":
        mlflow.lightgbm.autolog(
            log_models=False,
            log_model_signatures=False,
            log_input_examples=True,
            log_datasets=False,
            silent=True,
            disable=True
        )
    elif flavor == "catboost":
        pass

    return y_train, y_valid, run_name, flavor


def run_feature_selection_experiment(
    models: List,
    min_features: int,
    max_features: int,
    experiment_id: str,
    X_train: np.ndarray,
    y_train: np.array,
    X_valid: np.ndarray,
    y_valid: np.array,
    FEATURES_NAME: List[str],
    PROCESSED_RAW_FILE_PATH: str,
    ARTIFACTS_OUTPUT_PATH: str,
    FEATURES_OUTPUT_PATH: str,
    SEED: int
) -> None:
    """Run feature selection experiments for all models."""
    for model in models:
        new_y_train, new_y_valid, run_name, flavor = set_configurations_mlflow(
            model=model,
            y_train=y_train,
            y_valid=y_valid
        )
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            logger.info(f"[{datetime.now()}] Starting the run for the {run_name} model!\n")

            for i, n_features in enumerate(range(min_features, max_features + 1)):
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"{run_name}_experiment_{i}",
                    nested=True
                ):
                    logger.info(f"[{datetime.now()}] {run_name} - Experiment {i+1}/{max_features-min_features+1}: Testing {n_features} features...")
                    
                    # Create a fresh model instance for this experiment
                    if isinstance(model, DecisionTreeClassifier):
                        fresh_model = DecisionTreeClassifier(random_state=SEED)
                    elif isinstance(model, RandomForestClassifier):
                        fresh_model = RandomForestClassifier(random_state=SEED, verbose=0, n_jobs=-1)
                    elif isinstance(model, XGBClassifier):
                        fresh_model = XGBClassifier(random_state=SEED, n_jobs=-1)
                    elif isinstance(model, LGBMClassifier):
                        fresh_model = LGBMClassifier(random_state=SEED, verbose=-1, objective="multiclass")
                    elif isinstance(model, CatBoostClassifier):
                        fresh_model = CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)
                    else:
                        fresh_model = model
                    
                    results = apply_feature_selection(
                        model=fresh_model,
                        number_features=n_features,
                        X_train=X_train,
                        y_train=new_y_train,
                        X_valid=X_valid,
                        y_valid=new_y_valid,
                        FEATURES_NAME=FEATURES_NAME
                    )

                    # Log the trained model
                    if flavor == "sklearn":
                        mlflow.sklearn.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"]
                        )
                        mlflow.log_params(results["model"].get_params(deep=True))
                    elif flavor == "xgboost":
                        mlflow.xgboost.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"]
                        )
                        mlflow.log_params(results["model"].get_params(deep=True))
                    elif flavor == "lightgbm":
                        mlflow.lightgbm.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"]
                        )
                        mlflow.log_params(results["model"].get_params())
                    elif flavor == "catboost":
                        mlflow.catboost.log_model(
                            results["model"],
                            run_name,
                            signature=results["model_signature"]
                        )
                        mlflow.log_params(results["model"].get_all_params())

                    mlflow.log_metric("train_f1", results["train_f1"])
                    mlflow.log_metric("valid_f1", results["valid_f1"])
                    mlflow.log_artifact(PROCESSED_RAW_FILE_PATH)
                    mlflow.log_artifact(ARTIFACTS_OUTPUT_PATH)
                    mlflow.log_artifact(FEATURES_OUTPUT_PATH)
                    mlflow.log_param("features", results["features"])
                    
                    logger.info(f"[{datetime.now()}] ✓ {run_name}_experiment_{i} completed - Valid F1: {results['valid_f1']:.4f}")


class Objective:
    """Objective function for Optuna hyperparameter optimization."""
    
    def __init__(
        self,
        run_name: str,
        experiment_id: str,
        X_train: np.ndarray,
        y_train: np.array,
        X_valid: np.ndarray,
        y_valid: np.array,
        indexes: List,
        FEATURES_NAME: List[str],
        SEED: int
    ) -> None:
        self.run_name = run_name
        self.experiment_id = experiment_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.indexes_name = indexes
        self.indexes = [FEATURES_NAME.index(i) for i in indexes]
        self.SEED = SEED

        if self.run_name in ["decision_tree", "lightgbm", "catboost"]:
            self.y_train = np.argmax(self.y_train, axis=1)
            self.y_valid = np.argmax(self.y_valid, axis=1)
        
        self.X_train = self.X_train[:, self.indexes]
        self.X_valid = self.X_valid[:, self.indexes]
    
    def __call__(
        self,
        trial: optuna.trial.Trial
    ) -> float:
        with mlflow.start_run(experiment_id=self.experiment_id, nested=True):
            if self.run_name == "decision_tree":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 32, step=2),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 8, step=1),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6, step=1),
                    "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0, 0.5, step=0.1),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 16, step=2),
                    "random_state": self.SEED
                }
                model = DecisionTreeClassifier(**params)
            
            if self.run_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 10, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
                    "random_state": self.SEED,
                    "n_jobs": -1,
                }
                model = RandomForestClassifier(**params)
            
            if self.run_name == "xgboost":
                params = {
                    "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                    "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                    "random_state": self.SEED,
                    "n_jobs": -1,
                }
                model = XGBClassifier(**params)
            
            if self.run_name == "lightgbm":
                params = {
                    "objective": "multiclass",
                    "verbosity": -1,
                    "random_state": self.SEED,
                    "n_jobs": -1,
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                }
                model = LGBMClassifier(**params)
            
            if self.run_name == "catboost":
                params = {
                    "random_seed": self.SEED,
                    "verbose": 0,
                    "allow_writing_files": False,
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                    "depth": trial.suggest_int("depth", 1, 12),
                    "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                    "bootstrap_type": trial.suggest_categorical(
                        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                    )
                }
                model = CatBoostClassifier(**params)
            
            model.fit(X=self.X_train, y=self.y_train)

            train_prediction = model.predict(self.X_train)
            train_f1 = f1_score(
                y_true=self.y_train,
                y_pred=train_prediction,
                average="weighted"
            )

            valid_prediction = model.predict(self.X_valid)
            valid_f1 = f1_score(
                y_true=self.y_valid,
                y_pred=valid_prediction,
                average="weighted"
            )

            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("valid_f1", valid_f1)

            signature = infer_signature(
                model_input=self.X_train,
                model_output=train_prediction
            )

            if self.run_name in ["decision_tree", "random_forest"]:
                mlflow.sklearn.log_model(
                    model,
                    self.run_name,
                    signature=signature
                )
                mlflow.log_params(model.get_params(deep=True))
            elif self.run_name == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    self.run_name,
                    signature=signature
                )
                mlflow.log_params(model.get_params())
            elif self.run_name == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    self.run_name,
                    signature=signature
                )
                mlflow.log_params(model.get_params())
            elif self.run_name == "catboost":
                mlflow.catboost.log_model(
                    model,
                    self.run_name,
                    signature=signature
                )
                mlflow.log_params(model.get_all_params())

        return valid_f1


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info(f"Starting ML Experiments - {datetime.now()}")
    logger.info("=" * 80)
    
    # Configuration
    CONFIG_PATH = os.path.join("..", "src", "config")
    
    # Read version
    version_file = os.path.join(script_dir, "VERSION")
    with open(version_file, "r") as f:
        CODE_VERSION = f.readline().strip()
    
    # Load configuration files
    credentials_config = read_yaml_file(
        path=CONFIG_PATH,
        file="credentials.yaml"
    )
    
    general_settings = read_yaml_file(
        path=CONFIG_PATH,
        file="settings.yaml"
    )
    
    # Setup MLflow tracking
    if credentials_config["EC2"] != "YOUR_EC2_INSTANCE_URL":
        mlflow.set_tracking_uri(f"http://{credentials_config['EC2']}:5000")
    else:
        mlflow.set_tracking_uri(f"http://mlflow:5000") 
    
    logger.info(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")
    
    # Constants
    SEED = 42
    ARTIFACTS_OUTPUT_PATH = general_settings["ARTIFACTS_PATH"]
    FEATURES_OUTPUT_PATH = general_settings["FEATURES_PATH"]
    RAW_FILE_PATH = os.path.join(general_settings["DATA_PATH"], general_settings["RAW_FILE_NAME"])
    PROCESSED_RAW_FILE = "Preprocessed_" + general_settings["RAW_FILE_NAME"]
    PROCESSED_RAW_FILE_PATH = os.path.join(general_settings["DATA_PATH"], PROCESSED_RAW_FILE)
    FEATURE_SELECTION_EXPERIMENT_NAME = "feature-selection-experimentation"
    HYPERPARAMETER_TUNING_EXPERIMENT_NAME = "hyperparameters-tuning-experimentation"
    
    # Load data
    logger.info("\n[1/4] Loading data and artifacts...")
    X_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_train.pkl"))
    y_train = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_train.pkl"))
    X_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "X_valid.pkl"))
    y_valid = joblib.load(os.path.join(FEATURES_OUTPUT_PATH, "y_valid.pkl"))
    
    sc = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "features_sc.pkl"))
    ohe = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "features_ohe.pkl"))
    ohe_label = joblib.load(os.path.join(ARTIFACTS_OUTPUT_PATH, "label_ohe.pkl"))
    
    temp_df = pd.read_csv(PROCESSED_RAW_FILE_PATH, sep=",")
    FEATURES_NAME = temp_df.columns.tolist()
    del temp_df
    
    logger.info(f"  Loaded training data: {X_train.shape}")
    logger.info(f"  Loaded validation data: {X_valid.shape}")
    logger.info(f"  Total features: {len(FEATURES_NAME)}")
    
    # Create models
    logger.info("\n[2/4] Initializing models...")
    dt = DecisionTreeClassifier(random_state=SEED)
    rf = RandomForestClassifier(random_state=SEED, verbose=0, n_jobs=-1)
    xg = XGBClassifier(random_state=SEED, n_jobs=-1)
    lg = LGBMClassifier(random_state=SEED, verbose=-1, objective="multiclass")
    cb = CatBoostClassifier(random_seed=SEED, verbose=0, allow_writing_files=False)
    logger.info("  Models initialized")
    
    # Feature Selection Experiments
    logger.info("\n[3/4] Starting Feature Selection Experiments...")
    logger.info("-" * 80)
    models = [dt, rf, xg, lg]
    min_features = math.floor(X_train.shape[1] * 0.2)
    max_features = math.floor(X_train.shape[1] * 0.5)
    
    logger.info(f"  Testing {len(models)} models with {min_features}-{max_features} features each")
    logger.info(f"  Total experiments: {len(models)} models × {max_features - min_features + 1} feature counts")
    
    experiment_id = get_or_create_experiment(
        experiment_name=FEATURE_SELECTION_EXPERIMENT_NAME,
        tags={"version": "v2", "code_version": CODE_VERSION}
    )
    
    run_feature_selection_experiment(
        models=models,
        min_features=min_features,
        max_features=max_features,
        experiment_id=experiment_id,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        FEATURES_NAME=FEATURES_NAME,
        PROCESSED_RAW_FILE_PATH=PROCESSED_RAW_FILE_PATH,
        ARTIFACTS_OUTPUT_PATH=ARTIFACTS_OUTPUT_PATH,
        FEATURES_OUTPUT_PATH=FEATURES_OUTPUT_PATH,
        SEED=SEED
    )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Feature Selection Experiments Completed - {datetime.now()}")
    logger.info("=" * 80)
    
    # Hyperparameter Tuning Experiments
    logger.info("\n[4/4] Starting Hyperparameter Tuning Experiments...")
    logger.info("-" * 80)
    
    hpt_experiment_id = get_or_create_experiment(
        experiment_name=HYPERPARAMETER_TUNING_EXPERIMENT_NAME,
        tags={"version": "v2", "code_version": CODE_VERSION}
    )
    
    # Feature sets from feature selection experiments (these should be replaced with your actual best features)
    dt_features_indexes = ['Gender_x0_Male', 'Age_x0_q2', 'Age_x0_q4', 'CAEC_x0_no', 'SCC_x0_yes', 'MTRANS_x0_Motorbike', 'Weight', 'BMI', 'BSA', 'IBW', 'diff_W_IBW']
    rf_features_indexes = ['Gender_x0_Male', 'Age_x0_q4', 'family_history_with_overweight_x0_yes', 'FAVC_x0_yes', 'CAEC_x0_Frequently', 'CAEC_x0_no', 'SCC_x0_yes', 'MTRANS_x0_Public_Transportation', 'MTRANS_x0_Walking', 'HH_x0_good', 'Weight', 'FCVC', 'NCP', 'CH2O', 'TUE', 'BMI', 'IBW', 'BMR']
    xg_features_indexes = ['Gender_x0_Male', 'Age_x0_q2', 'Age_x0_q3', 'Age_x0_q4', 'family_history_with_overweight_x0_yes', 'FAVC_x0_yes', 'CAEC_x0_no', 'SCC_x0_yes', 'CALC_x0_no', 'MTRANS_x0_Bike', 'MTRANS_x0_Walking', 'SWC_x0_1', 'IS_x0_1', 'Weight', 'FCVC', 'NCP', 'FAF', 'BMI']
    lg_features_indexes = ['Gender_x0_Male', 'Age_x0_q3', 'Age_x0_q4', 'FAVC_x0_yes', 'CAEC_x0_Frequently', 'SCC_x0_yes', 'CALC_x0_no', 'EVEMM_x0_1', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'PAL', 'IBW']
    
    # Run hyperparameter tuning for each model
    hyperparameter_models = [
        ("decision_tree", dt_features_indexes),
        ("random_forest", rf_features_indexes),
        ("xgboost", xg_features_indexes),
        ("lightgbm", lg_features_indexes),
    ]
    
    for run_name, features_indexes in hyperparameter_models:
        logger.info(f"\n[{datetime.now()}] Starting hyperparameter tuning for {run_name}...")
        logger.info(f"  Using {len(features_indexes)} features")
        
        with mlflow.start_run(experiment_id=hpt_experiment_id, run_name=run_name):
            objective = Objective(
                run_name=run_name,
                experiment_id=hpt_experiment_id,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                y_valid=y_valid,
                indexes=features_indexes,
                FEATURES_NAME=FEATURES_NAME,
                SEED=SEED
            )
            
            study = optuna.create_study(direction="maximize")
            logger.info(f"  Running 100 trials for {run_name}...")
            study.optimize(objective, n_trials=100, show_progress_bar=True)
            
            logger.info(f"  ✓ {run_name} completed - Best F1: {study.best_value:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"ALL EXPERIMENTS COMPLETED - {datetime.now()}")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Open MLflow UI at http://localhost:5000")
    logger.info("2. Compare model performances")
    logger.info("3. Register the best model using the notebook's 'Registering Best Models' section")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n Training interrupted by user. Partial results may be available in MLflow.")
        sys.exit(1)
    except Exception as e:
        logger.info(f"\n\n Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

