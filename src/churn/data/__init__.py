"""Data module for churn project."""

from enum import StrEnum


class DataFile(StrEnum):
    CUSTOMER_CHURN = "telecom_customer_churn"
    DATA_DICTIONARY = "telecom_data_dictionary"
    ZIPCODE_POPULATION = "telecom_zipcode_population"


class DatasetSize(StrEnum):
    SMALL = "small"
    FULL = "full"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"


CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Contract",
    "Payment Method",
    "Offer",
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

ORDINAL_FEATURES = [
    "Age Rank",
    "Tenure Rank",
    "Dependents Rank",
    "Referrals Rank",
]
TARGET_VARIABLE = "Churn Status"
