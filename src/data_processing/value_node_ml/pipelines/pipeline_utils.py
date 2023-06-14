import pandas as pd
from typing import Optional

from value_node_ml.params.params import ProgramParams
from value_node_ml.data_loading.data_types import SamplesAndLabels
from commons.params.data_origin import DataOriginEnum
from sklearn.model_selection import train_test_split


def split_dataset_if_needed(
    samples_and_labels_train: SamplesAndLabels, 
    samples_and_labels_test: Optional[SamplesAndLabels]
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
    origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
):
    """
    Helper function for handling data origins, for classifiers that do not use data generators.
    """
    __samples: list[pd.DataFrame] = []
    __labels: list[pd.Series] = []
    for origin in data_origins:
        samples_and_labels = origin_to_samples_and_labels[origin]

        samples, labels = samples_and_labels
        __labels += [labels]
        __samples += [samples]
       
    samples_and_labels = (pd.concat(__samples), pd.concat(__labels))
    return samples_and_labels

def split_samples_and_labels(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
):
    """
    Split samples and labels into training and testing sets.
    """
    samples_and_labels_train = handle_data_origin(
        params.data_origins_training,
        origin_to_samples_and_labels
    )
    samples_and_labels_test = None
    if params.data_origins_testing is not None:
        samples_and_labels_test = handle_data_origin(
            params.data_origins_testing,
            origin_to_samples_and_labels
        )
    
    return samples_and_labels_train, samples_and_labels_test