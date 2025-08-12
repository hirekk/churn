"""Download data utilities."""

import logging
import os
from pathlib import Path

from churn.data import DataFile
from churn.logger import DEFAULT_LOGGER

DATASET_NAME = "shilongzhuang/telecom-customer-churn-by-maven-analytics"
DATASET_URL = (
    "https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics/data"
)


def should_download_dataset(
    data_dirpath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> bool:
    """Check if dataset download is needed."""
    missing_files = [
        str(file.absolute())
        for file in [
            data_dirpath / f"{DataFile.CUSTOMER_CHURN.value}.csv",
            data_dirpath / f"{DataFile.DATA_DICTIONARY.value}.csv",
            data_dirpath / f"{DataFile.ZIPCODE_POPULATION.value}.csv",
        ]
        if not file.exists()
    ]

    if missing_files:
        logger.warning(
            "Dataset download required; missing files: %s",
            ", ".join(missing_files),
        )
        return True

    logger.info("All dataset files exist; skipping download")
    return False


def create_kaggle_client(logger: logging.Logger = DEFAULT_LOGGER):  # noqa: ANN201
    """Create Kaggle API client.

    Raises:
        RuntimeError: Kaggle credentials are not set.
    """
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        errmsg = (
            "Kaggle credentials are not set; "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set "
            "or a .env file configuring them must exist in project root"
        )
        logger.warning(errmsg)
        raise RuntimeError(errmsg)

    # This import needs to be here because the kaggle package eagerly checks for
    # credentials and will raise an error at import time that may be confusing if
    # kaggle config file does not exist and environment variables are not set.
    # This helps to handle it more gracefully.
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    client = KaggleApi()
    client.authenticate()

    return client


def download_dataset(
    data_dirpath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> Path:
    """Download dataset from Kaggle."""
    if not should_download_dataset(data_dirpath, logger):
        return data_dirpath / f"{DataFile.CUSTOMER_CHURN.value}.csv"

    client = create_kaggle_client(logger)

    logger.info("Downloading %s dataset...", DATASET_NAME)
    client.dataset_download_files(
        DATASET_NAME,
        path=data_dirpath,
        unzip=True,
    )

    logger.info(
        "Dataset %s downloaded successfully to %s",
        DATASET_NAME,
        data_dirpath.absolute(),
    )

    return data_dirpath / f"{DataFile.CUSTOMER_CHURN.value}.csv"
