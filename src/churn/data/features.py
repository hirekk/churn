"""Feature engineering for the telecom churn project."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data import CATEGORICAL_FEATURES
from churn.data import TARGET_VARIABLE
from churn.logger import DEFAULT_LOGGER


def create_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ordinal features.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with ordinal features
    """
    age_rank = pd.cut(
        df["Age"],
        bins=[-np.inf, 25, 35, 50, 70, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    tenure_rank = pd.cut(
        df["Tenure in Months"],
        bins=[-np.inf, 2, 6, 12, 24, 60, np.inf],
        labels=[1, 2, 3, 4, 5, 6],
    )
    dependents_rank = pd.cut(
        df["Number of Dependents"],
        bins=[-np.inf, 0, 1, 3, np.inf],
        labels=[1, 2, 3, 4],
    )
    referrals_rank = pd.cut(
        df["Number of Referrals"],
        bins=[-np.inf, 0, 1, 3, 10, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return pd.DataFrame({
        "age_rank": age_rank,
        "tenure_rank": tenure_rank,
        "dependents_rank": dependents_rank,
        "referrals_rank": referrals_rank,
    })


def select_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select categorical features.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with categorical features
    """
    return df[CATEGORICAL_FEATURES]


def create_referrals_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create referrals features.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with referrals features
    """
    num_referrals = df["Number of Referrals"]

    referrals_per_year = (num_referrals / df["Tenure in Months"] * 12).fillna(0)
    referrals_per_revenue_thousands = (num_referrals / df["Total Charges"] * 1000).fillna(0)

    return pd.DataFrame(
        data={
            "num_referrals": num_referrals,
            "referrals_per_year": referrals_per_year,
            "referrals_per_revenue_thousands": referrals_per_revenue_thousands,
        },
        index=df.index,
    )


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count total add-on services for each customer.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with service features
    """
    addon_services = [
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
        "Paperless Billing",
    ]

    return df[addon_services].eq("Yes").astype(int).sum(axis=1).rename("addon_services_count")


def create_billing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price per GB for internet customers.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with billing features
    """
    price_per_gb = pd.Series(
        data=df["Monthly Charge"] / df["Avg Monthly GB Download"],
        index=df.index,
        name="price_per_gb",
    )
    charge_per_revenue = pd.Series(
        data=df["Total Charges"] / df["Total Revenue"],
        index=df.index,
        name="charge_per_revenue",
    )

    monthly_charge = df["Monthly Charge"]
    monthly_refunds = pd.Series(
        data=df["Total Refunds"] / df["Tenure in Months"],
        index=df.index,
        name="monthly_refunds",
    )
    refunds_per_revenue = pd.Series(
        data=df["Total Refunds"] / df["Total Revenue"],
        index=df.index,
        name="refunds_per_revenue",
    )
    monthly_extra_data_charges = pd.Series(
        data=df["Total Extra Data Charges"] / df["Tenure in Months"],
        index=df.index,
        name="monthly_extra_data_charges",
    )
    extra_data_charges_per_revenue = pd.Series(
        data=df["Total Extra Data Charges"] / df["Total Revenue"],
        index=df.index,
        name="extra_data_charges_per_revenue",
    )
    monthly_long_distance_charges = pd.Series(
        data=df["Total Long Distance Charges"] / df["Tenure in Months"],
        index=df.index,
        name="monthly_long_distance_charges",
    )
    long_distance_charges_per_revenue = pd.Series(
        data=df["Total Long Distance Charges"] / df["Total Revenue"],
        index=df.index,
        name="long_distance_charges_per_revenue",
    )
    return pd.concat(
        [
            price_per_gb,
            charge_per_revenue,
            monthly_charge,
            monthly_refunds,
            refunds_per_revenue,
            monthly_extra_data_charges,
            extra_data_charges_per_revenue,
            monthly_long_distance_charges,
            long_distance_charges_per_revenue,
        ],
        axis=1,
    )


def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create geographic features from coordinates.

    Args:
        df: Raw dataframe

    Returns:
        Dataframe with geographic features
    """
    ca_center_lat, ca_center_lon = 36.7783, -119.4179

    distance_from_center = pd.Series(
        data=np.sqrt(
            (df["Latitude"] - ca_center_lat) ** 2 + (df["Longitude"] - ca_center_lon) ** 2,
        ),
        index=df.index,
        name="distance_from_center",
    )

    region = pd.Series(
        data=pd.cut(
            df["Latitude"],
            bins=[float("-inf"), 35, 40, float("inf")],
            labels=["southern", "central", "northern"],
            include_lowest=True,
        ),
        index=df.index,
        name="region",
    )
    region_one_hot = pd.get_dummies(
        region,
        prefix="region",
        drop_first=True,
    ).astype(int)

    return pd.concat([distance_from_center, region_one_hot], axis=1)


def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target variable for churn prediction.

    Args:
        df: Raw dataframe

    Returns:
        Series with target variable
    """
    return pd.Series(
        data=(df["Customer Status"] == "Churned").astype(int),
        index=df.index,
        name=TARGET_VARIABLE,
    )


def create_features(
    df: pd.DataFrame,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Args:
        df: Raw dataframe
        logger: Logger instance

    Returns:
        Dataframe with engineered features
    """
    logger.info("Starting feature engineering...")

    ordinal_features = create_ordinal_features(df)
    categorical_features = select_categorical_features(df)
    referrals_features = create_referrals_features(df)
    service_features = create_service_features(df)
    geographic_features = create_geographic_features(df)
    billing_features = create_billing_features(df)

    feat_df = pd.concat(
        [
            ordinal_features,
            categorical_features,
            referrals_features,
            service_features,
            geographic_features,
            billing_features,
        ],
        axis=1,
    )

    logger.info("Feature engineering completed")
    logger.info("Final dataset shape: %s", feat_df.shape)

    return feat_df


def create_features_with_target(
    df: pd.DataFrame,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> pd.DataFrame:
    """Apply feature engineering and include target variable (for training only).

    Args:
        df: Raw dataframe
        logger: Logger instance

    Returns:
        Dataframe with engineered features and target variable
    """
    logger.info("Starting feature engineering with target...")
    features = create_features(df, logger)
    target = create_target_variable(df)

    result = pd.concat([features, target], axis=1)
    logger.info("Feature engineering with target completed")
    logger.info("Final dataset shape: %s", result.shape)

    return result


def make_dataset(
    input_filepath: Path,
    output_filepath: Path,
    logger: logging.Logger = DEFAULT_LOGGER,
) -> pd.DataFrame:
    """Complete pipeline: load data, engineer features, save for modeling.

    Args:
        input_filepath: Path to raw CSV file
        output_filepath: Path to save processed CSV
        logger: Logger instance

    Returns:
        Processed dataframe ready for ML
    """
    logger.info("Loading data from %s", input_filepath.absolute())
    df = pd.read_csv(input_filepath)

    feat_df = create_features_with_target(df, logger)

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_filepath, index=False)
    logger.info("Saved ML-ready dataset to %s", output_filepath.absolute())

    return feat_df
