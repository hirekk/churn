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
