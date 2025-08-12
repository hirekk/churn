"""Feature engineering for the telecom churn project."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churn.logger import DEFAULT_LOGGER


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to lowercase and replace spaces with underscores."""
    return df.rename(columns=lambda name: name.lower().replace(" ", "_"))


def create_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age buckets."""
    age_rank = pd.cut(
        df["age"],
        bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    num_dependents_rank = pd.cut(
        df["number_of_dependents"],
        bins=[-np.inf, 0, 1, 3, np.inf],
        labels=[0, 1, 2, 3],
    )

    referrals_rank = pd.cut(
        df["number_of_referrals"],
        bins=[-np.inf, 0, 1, 5, 10, np.inf],
        labels=[0, 1, 2, 3, 4],
    )

    tenure_rank = pd.cut(
        df["tenure_in_months"],
        bins=[0, 6, 12, 24, float("inf")],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    )

    return pd.DataFrame(
        data={
            "age_rank": age_rank,
            "num_dependents_rank": num_dependents_rank,
            "referrals_rank": referrals_rank,
            "tenure_rank": tenure_rank,
        },
        index=df.index,
    ).astype(int)


def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create categorical features."""
    return pd.concat(
        [
            pd.get_dummies(
                df[["gender", "married", "contract", "payment_method"]], drop_first=True
            ).astype(int),
            pd.get_dummies(df["offer"]).astype(int),
            pd.get_dummies(df[["internet_type"]]).astype(int),
        ],
        axis=1,
    )


def create_referrals_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create referrals features."""
    num_referrals = df["number_of_referrals"]

    referrals_per_year = (num_referrals / df["tenure_in_months"] * 12).fillna(0)
    referrals_per_revenue_thousands = (num_referrals / df["total_charges"] * 1000).fillna(0)

    return pd.DataFrame(
        data={
            "num_referrals": num_referrals,
            "referrals_per_year": referrals_per_year,
            "referrals_per_revenue_thousands": referrals_per_revenue_thousands,
        },
        index=df.index,
    )


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count total add-on services for each customer."""
    addon_services = [
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection_plan",
        "premium_tech_support",
        "streaming_tv",
        "streaming_movies",
        "streaming_music",
        "unlimited_data",
        "paperless_billing",
    ]

    feat_df = df[addon_services].eq("Yes").astype(int)

    feat_df = pd.concat([feat_df, feat_df.sum(axis=1).rename("addon_services_count")], axis=1)

    return feat_df.rename(columns=lambda name: name.lower().replace(" ", "_"))


def create_billing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price per GB for internet customers."""
    price_per_gb = pd.Series(
        data=df["monthly_charge"] / df["avg_monthly_gb_download"],
        index=df.index,
        name="price_per_gb",
    )
    charge_per_revenue = pd.Series(
        data=df["total_charges"] / df["total_revenue"],
        index=df.index,
        name="charge_per_revenue",
    )

    monthly_charge = df["monthly_charge"]
    monthly_refunds = pd.Series(
        data=df["total_refunds"] / df["tenure_in_months"],
        index=df.index,
        name="monthly_refunds",
    )
    refunds_per_revenue = pd.Series(
        data=df["total_refunds"] / df["total_revenue"],
        index=df.index,
        name="refunds_per_revenue",
    )
    monthly_extra_data_charges = pd.Series(
        data=df["total_extra_data_charges"] / df["tenure_in_months"],
        index=df.index,
        name="monthly_extra_data_charges",
    )
    extra_data_charges_per_revenue = pd.Series(
        data=df["total_extra_data_charges"] / df["total_revenue"],
        index=df.index,
        name="extra_data_charges_per_revenue",
    )
    monthly_long_distance_charges = pd.Series(
        data=df["total_long_distance_charges"] / df["tenure_in_months"],
        index=df.index,
        name="monthly_long_distance_charges",
    )
    long_distance_charges_per_revenue = pd.Series(
        data=df["total_long_distance_charges"] / df["total_revenue"],
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
    """Create geographic features from coordinates."""
    # Create distance from center (rough approximation)
    # Using approximate center of California
    ca_center_lat, ca_center_lon = 36.7783, -119.4179

    distance_from_center = pd.Series(
        data=np.sqrt(
            (df["latitude"] - ca_center_lat) ** 2 + (df["longitude"] - ca_center_lon) ** 2
        ),
        index=df.index,
        name="distance_from_center",
    )

    # Create region based on latitude
    region = pd.Series(
        data=pd.cut(
            df["latitude"],
            bins=[float("-inf"), 35, 40, float("inf")],
            labels=["southern", "central", "northern"],
            include_lowest=True,
        ),
        index=df.index,
        name="region",
    )
    region_one_hot = pd.get_dummies(region, prefix="region", drop_first=True).astype(int)

    return pd.concat([distance_from_center, region_one_hot], axis=1)


def create_target_variable(df: pd.DataFrame) -> pd.Series:
    """Create binary target variable for churn prediction."""
    # Create binary target: 1 = Churned, 0 = Stayed
    return pd.Series(
        data=(df["customer_status"] == "Churned").astype(int),
        index=df.index,
        name="target",
    )


def create_features(df: pd.DataFrame, logger: logging.Logger = DEFAULT_LOGGER) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Args:
        df: Raw dataframe
        logger: Logger instance

    Returns:
        Dataframe with engineered features
    """
    logger.info("Starting feature engineering...")
    df = rename_columns(df)

    # Create target variable first
    target = create_target_variable(df)
    logger.info("Created target variable. Dataset shape: %s", df.shape)

    # Apply feature engineering transformations
    ordinal_features = create_ordinal_features(df)
    categorical_features = create_categorical_features(df)
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
            target,
        ],
        axis=1,
    ).rename(columns=lambda name: name.lower().replace(" ", "_"))

    logger.info("Feature engineering completed")
    logger.info("Final dataset shape: %s", feat_df.shape)

    return feat_df


def make_dataset(
    input_filepath: Path, output_filepath: Path, logger: logging.Logger = DEFAULT_LOGGER
) -> pd.DataFrame:
    """Complete pipeline: load data, engineer features, save for modeling.

    Args:
        input_filepath: Path to raw CSV file
        output_filepath: Path to save processed CSV
        logger: Logger instance

    Returns:
        Processed dataframe ready for ML
    """
    # Load data
    logger.info("Loading data from %s", input_filepath.absolute())
    df = pd.read_csv(input_filepath)

    # Engineer features
    feat_df = create_features(df, logger)

    # Save processed data
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_filepath, index=False)
    logger.info("Saved ML-ready dataset to %s", output_filepath.absolute())

    return feat_df
