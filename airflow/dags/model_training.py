"""Airflow DAG for telecom customer churn data extraction."""

from pathlib import Path

from churn.data import DataFile
from churn.data import DatasetSize
from churn.data import DatasetSplit
from churn.data.download import download_dataset
from churn.data.feature_engineering import make_dataset
from churn.data.load import load_data
from churn.data.preprocess import extract_small_sample
from churn.data.preprocess import remove_customer_category_joined
from churn.logger import DEFAULT_LOGGER
from churn.model.train import evaluate_model
from churn.model.train import train_random_forest
import mlflow

from airflow import DAG
from airflow.sdk import task

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
    """Download dataset if it does not exist."""
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
def train_model(
    experiment_id: str = "churn_prediction",
    random_state: int = 42,
) -> None:
    """Train churn prediction model with MLflow tracking."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_id)

    with mlflow.start_run() as run:
        DEFAULT_LOGGER.info(f"Starting MLflow run {run.info}")

        # Log basic experiment info
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("random_state", random_state)

        # Load data
        X, y = load_data(split=DatasetSplit.TRAIN, data_dirpath=DATA_PROCESSED_DIR)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_dict(y.value_counts().to_dict(), "target_distribution.json")

        # Train model
        model = train_random_forest(X, y, random_state=random_state)
        mlflow.log_model_params(model.get_params(), "random_forest_params.json")

        # Evaluate model
        metrics = evaluate_model(model, X, y)
        mlflow.log_dict(metrics, "metrics.json")

        # Save model locally for Docker deployment
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_dirpath = models_dir / "random_forest"
        model_dirpath.mkdir(exist_ok=True)

        # Save model using MLflow
        import mlflow.sklearn

        mlflow.sklearn.save_model(model, model_dirpath)

        # Save feature names for preprocessing
        import json

        feature_names = list(X.columns)
        with open(model_dirpath / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f)

        DEFAULT_LOGGER.info(f"Model saved locally to {model_dirpath}")
        DEFAULT_LOGGER.info(f"Model files: {list(model_dirpath.glob('*'))}")


with DAG(
    "model_training",
    description="Train churn prediction model",
    schedule=None,
    catchup=False,
    tags=["churn", "model-training"],
) as dag:
    _ = extract_dataset() >> make_features() >> train_model()
