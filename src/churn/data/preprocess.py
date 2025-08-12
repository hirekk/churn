"""Data preprocessing functions for the telecom churn project."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.logger import DEFAULT_LOGGER


def remove_customer_category_joined(
    input_filepath: Path,
    output_filepath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> Path:
    """Remove customers who are in the 'Joined' category.

    Args:
        input_filepath: Path to the raw CSV file
        output_filepath: Path to save the dataset without joined customers
        logger: Logger instance

    Returns:
        Path to the dataset without 'Joined' customers category
    """
    df = pd.read_csv(input_filepath)
    logger.info("Loaded raw dataset from %s", input_filepath.absolute())
    logger.debug("Raw dataset shape: %s", df.shape)

    joined_mask = df["Customer Status"] == "Joined"
    df = df.loc[~joined_mask, :].copy()
    logger.debug("Dataset shape after removing joined customers: %s", df.shape)

    df.to_csv(output_filepath, index=False)
    logger.info(
        "Saved dataset without category 'Joined' customers to %s",
        output_filepath.absolute(),
    )

    return output_filepath


def extract_small_sample(
    input_filepath: Path,
    output_filepath: Path,
    sample_size: int = 1000,
    random_state: int = 42,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> Path:
    """Extract a small sample from the raw dataset.

    Args:
        input_filepath: Path to the raw CSV file
        output_filepath: Path to save the small sample
        sample_size: Number of rows to sample (default: 1000)
        random_state: Random seed for reproducibility
        logger: Logger instance

    Returns:
        Path to the small sample dataset
    """
    df_full = pd.read_csv(input_filepath)
    logger.info("Loaded raw dataset from %s", input_filepath.absolute())
    logger.debug("Raw dataset shape: %s", df_full.shape)

    df_small = df_full.sample(n=sample_size, random_state=random_state)
    logger.debug("Sampled dataset shape: %s", df_small.shape)

    df_small.to_csv(output_filepath, index=False)
    logger.info(
        "Extracted small sample from %s to %s",
        input_filepath.absolute(),
        output_filepath.absolute(),
    )

    return output_filepath


def split_data(
    input_filepath: Path,
    output_dirpath: Path,
    test_size: float = 0.2,
    stratify_by: str | None = None,
    random_state: int = 42,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> tuple[Path, Path]:
    """Split raw data into training and test datasets.

    Args:
        input_filepath: Path to the raw CSV file
        output_dirpath: Directory to save processed datasets
        test_size: Proportion of data to use for testing
        stratify_by: Column to use for stratification
        random_state: Random seed for reproducibility
        logger: Logger instance

    Returns:
        Path to the training dataset
        Path to the test dataset
    """
    df = pd.read_csv(input_filepath)

    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_by] if stratify_by else None,
    )
    train_df = pd.DataFrame(train_data, columns=df.columns)
    logger.info("Training dataset shape: %s", train_df.shape)
    test_df = pd.DataFrame(test_data, columns=df.columns)
    logger.info("Test dataset shape: %s", test_df.shape)

    train_dirpath = output_dirpath / "train"
    train_dirpath.mkdir(parents=True, exist_ok=True)
    test_dirpath = output_dirpath / "test"
    test_dirpath.mkdir(parents=True, exist_ok=True)

    train_filepath = train_dirpath / input_filepath.name
    test_filepath = test_dirpath / input_filepath.name

    train_df.to_csv(train_filepath, index=False)
    logger.info("Saved training dataset to %s", train_filepath.absolute())
    test_df.to_csv(test_filepath, index=False)
    logger.info("Saved test dataset to %s", test_filepath.absolute())

    return train_filepath, test_filepath
