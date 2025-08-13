"""Airflow DAG for telecom customer churn data extraction."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from airflow import DAG
from airflow.sdk import task
import mlflow
import mlflow.sklearn
import pandas as pd

from churn.data import TARGET_VARIABLE
from churn.data import DataFile
from churn.data import DatasetSize
from churn.data import DatasetSplit
from churn.data.download import download_dataset
from churn.data.features import make_dataset
from churn.data.load import load_data
from churn.data.preprocess import extract_small_sample
from churn.data.preprocess import remove_customer_category_joined
from churn.data.preprocess import split_data
from churn.logger import DEFAULT_LOGGER
from churn.model.test import evaluate_test_performance
from churn.model.train import evaluate_model
from churn.model.train import train_random_forest


if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@task
def extract_dataset(
    exclude_joined: bool = False,
    size: DatasetSize = DatasetSize.FULL,
) -> None:
    """Download dataset if it does not exist.

    Args:
        exclude_joined: Whether to exclude joined customers
        size: Size of the dataset to extract
    """
    # Access runtime configuration from dag_run
    # exclude_joined = dag_run.conf.get('exclude_joined', exclude_joined)
    # size_str = dag_run.conf.get('size', size.value)

    data_path = download_dataset(data_dirpath=DATA_RAW_DIR)
    if exclude_joined:
        data_path = remove_customer_category_joined(
            input_filepath=data_path,
            output_filepath=DATA_INTERIM_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv",
        )
    if size == DatasetSize.SMALL:
        data_path = extract_small_sample(
            input_filepath=data_path,
            output_filepath=DATA_INTERIM_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv",
        )

    # Expectation is after this task, a dataset is present in the interim data directory
    # for the subsequent feature engineering task.
    # If neither of the two above transformations is applied, the dataset needs to be
    # moved from the raw data directory to the expected location.
    if data_path.parent == DATA_RAW_DIR:
        data_path = data_path.rename(DATA_INTERIM_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv")


@task
def make_features() -> None:
    """Apply feature engineering to prepare ML-ready dataset."""
    make_dataset(
        input_filepath=DATA_INTERIM_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv",
        output_filepath=DATA_PROCESSED_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv",
    )


@task
def split_dataset() -> None:
    """Split dataset into training and test sets."""
    split_data(
        input_filepath=DATA_PROCESSED_DIR / f"{DataFile.CUSTOMER_CHURN.value}.csv",
        output_dirpath=DATA_PROCESSED_DIR,
        test_size=0.2,
        stratify_by=TARGET_VARIABLE,
        random_state=42,
    )


@task
def train_model(
    experiment_id: str = "churn_prediction",
    random_state: int = 42,
) -> str:
    """Train churn prediction model with MLflow tracking.

    Args:
        experiment_id: ID of the experiment to use
        random_state: Random state for reproducibility

    Returns:
        ID of the run
    """
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("random_state", random_state)

        X, y = load_data(split=DatasetSplit.TRAIN, data_dirpath=DATA_PROCESSED_DIR)
        mlflow.log_param("train_num_samples", len(X))
        mlflow.log_param("train_num_features", len(X.columns))
        mlflow.log_dict(y.value_counts().to_dict(), "train_target_distribution.json")

        model = train_random_forest(X, y, random_state=random_state)

        evaluate_model(model, X, y)

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_dirpath = models_dir / "random_forest"
        model_dirpath.mkdir(exist_ok=True)

        train_metrics = evaluate_model(model, X, y)
        mlflow.log_metrics({
            "cv_f1_mean": train_metrics["cv_f1_mean"],
            "cv_f1_std": train_metrics["cv_f1_std"],
            "cv_precision_mean": train_metrics["cv_precision_mean"],
            "cv_precision_std": train_metrics["cv_precision_std"],
            "cv_recall_mean": train_metrics["cv_recall_mean"],
            "cv_recall_std": train_metrics["cv_recall_std"],
            "cv_pr_auc_mean": train_metrics["cv_pr_auc_mean"],
            "cv_pr_auc_std": train_metrics["cv_pr_auc_std"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "pr_auc": train_metrics["pr_auc"],
        })

        feature_importance_file = model_dirpath / "train_feature_importance.csv"
        train_metrics["feature_importance"].to_csv(feature_importance_file, index=False)
        mlflow.log_artifact(str(feature_importance_file))

        confusion_matrix_file = model_dirpath / "train_confusion_matrix.csv"
        cm_df = pd.DataFrame(
            data=train_metrics["confusion_matrix"],
            columns=pd.Index(["Predicted_0", "Predicted_1"]),
            index=pd.Index(["Actual_0", "Actual_1"]),
        )
        cm_df.to_csv(confusion_matrix_file, index=True)
        mlflow.log_artifact(str(confusion_matrix_file))

        mlflow.sklearn.save_model(model, model_dirpath / run.info.run_id)

        feature_names = list(X.columns)
        feature_names_filepath = model_dirpath / "feature_names.json"
        with feature_names_filepath.open(mode="w", encoding="utf-8") as f:
            json.dump(feature_names, f)

        DEFAULT_LOGGER.info(f"Model saved locally to {model_dirpath}")
        DEFAULT_LOGGER.info(f"Model files: {list(model_dirpath.glob('*'))}")

    return run.info.run_id


@task
def test_model(
    run_id: Any,  # Accepts any type (including XCom values)
    experiment_id: str = "churn_prediction",
) -> None:
    """Test trained model on test dataset with MLflow tracking.

    Args:
        run_id: ID of the run to test
        experiment_id: ID of the experiment to use
    """
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run(run_id=run_id, nested=True):
        X_test, y_test = load_data(split=DatasetSplit.TEST, data_dirpath=DATA_PROCESSED_DIR)
        mlflow.log_param("test_num_samples", len(X_test))
        mlflow.log_param("test_num_features", len(X_test.columns))
        mlflow.log_dict(y_test.value_counts().to_dict(), "test_target_distribution.json")

        model_run_path = Path("models", "random_forest", run_id)
        model: Pipeline = mlflow.sklearn.load_model(str(model_run_path))

        test_metrics = evaluate_test_performance(model, X_test, y_test, model_run_path)
        mlflow.log_metrics({
            "test_f1": test_metrics["test_f1"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
            "test_pr_auc": test_metrics["test_pr_auc_avg"],
        })

        test_confusion_matrix_file = model_run_path / "test_confusion_matrix.csv"
        cm_df = pd.DataFrame(
            data=test_metrics["test_confusion_matrix"],
            columns=pd.Index(["Predicted_0", "Predicted_1"]),
            index=pd.Index(["Actual_0", "Actual_1"]),
        )
        cm_df.to_csv(test_confusion_matrix_file, index=True)
        mlflow.log_artifact(str(test_confusion_matrix_file))

        test_classification_report_file = model_run_path / "test_classification_report.csv"
        report_df = pd.DataFrame(test_metrics["test_classification_report"]).transpose()
        report_df.to_csv(test_classification_report_file)
        mlflow.log_artifact(str(test_classification_report_file))


with DAG(
    "model_training",
    description="Train churn prediction model",
    schedule=None,
    catchup=False,
    tags=["churn", "model-training"],
) as dag:
    train_result = train_model()
    _ = (
        extract_dataset()
        >> make_features()
        >> split_dataset()
        >> train_result
        >> test_model(run_id=train_result)
    )
