import pandas as pd
from typing import Optional, Any

from value_node_ml.params.params import ProgramParams
from value_node_ml.data_loading.data_types import PreprocessedData
from commons.params.data_origin import DataOriginEnum
from sklearn.model_selection import train_test_split


def split_dataset_if_needed(
    samples_and_labels_train: PreprocessedData, 
    samples_and_labels_test: Optional[PreprocessedData]
):
    """
    Split data into training and test sets if needed.
    NOTE: Needed when no testing data is provided).
    """
    if samples_and_labels_test is None:
        # Split data into training and test sets
        __samples, __labels = samples_and_labels_train
        X_train, X_test, y_train, y_test = train_test_split(__samples, __labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = samples_and_labels_train
        X_test, y_test = samples_and_labels_test

    return X_train, X_test, y_train, y_test


def handle_data_origin(
    data_origins: set[DataOriginEnum],
    origin_to_preprocessed_data: dict[DataOriginEnum, PreprocessedData]
):
    """
    Helper function for handling data origins.
    """
    __samples: list[pd.DataFrame] = []
    __samples_str: list[pd.DataFrame] = []
    __labels: list[pd.Series] = []
    for origin in data_origins:
        preprocessed_data = origin_to_preprocessed_data[origin]

        samples, samples_str, labels = preprocessed_data
        __samples += [samples]
        __samples_str += [samples_str]
        __labels += [labels]
       
    preprocessed_data = (pd.concat(__samples), pd.concat(__samples_str), pd.concat(__labels))
    return preprocessed_data


def split_preprocessed_data_by_origin(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, PreprocessedData]
):
    """
    Split samples and labels into training and testing sets.
    """
    preprocessed_data_train = handle_data_origin(
        params.data_origins_training,
        origin_to_samples_and_labels
    )
    preprocessed_data_test = None
    if params.data_origins_testing is not None:
        preprocessed_data_test = handle_data_origin(
            params.data_origins_testing,
            origin_to_samples_and_labels
        )
    
    return preprocessed_data_train, preprocessed_data_test


def keep_only_samples_and_labels(
    preprocessed_data: Optional[PreprocessedData]
):
    """
    Keep only the samples and labels from the preprocessed data.
    """
    if preprocessed_data is not None:
        return preprocessed_data[0], preprocessed_data[2]
    else:
        return None