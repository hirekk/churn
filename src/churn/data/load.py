"""Load processed dataset and split into features and target."""

import logging
from pathlib import Path

import pandas as pd

from churn.data import DataFile
from churn.data import DatasetSplit
from churn.logger import DEFAULT_LOGGER


def load_data(
    split: DatasetSplit,
    data_dirpath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed dataset and split into features and target."""
    data_path = data_dirpath / split.value / f"{DataFile.CUSTOMER_CHURN.value}.csv"
    logger.info("Loading data from %s", data_path.absolute())

    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    logger.info("Loaded %s samples with %s features", len(X), len(X.columns))
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    return X, y
