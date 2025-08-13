"""Load processed dataset and split into features and target."""

import logging
from pathlib import Path

import pandas as pd

from churn.data import TARGET_VARIABLE
from churn.data import DataFile
from churn.data import DatasetSplit
from churn.logger import DEFAULT_LOGGER


def load_data(
    split: DatasetSplit,
    data_dirpath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed dataset and split into features and target.

    Args:
        split: Dataset split to load
        data_dirpath: Path to data directory
        logger: Logger instance

    Returns:
        Tuple containing features dataframe and target variable series
    """
    data_path = data_dirpath / split.value / f"{DataFile.CUSTOMER_CHURN.value}.csv"
    logger.info("Loading data from %s", data_path.absolute())

    df = pd.read_csv(data_path)

    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    logger.info("Loaded %s samples with %s features", len(X), len(X.columns))
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    return X, y
