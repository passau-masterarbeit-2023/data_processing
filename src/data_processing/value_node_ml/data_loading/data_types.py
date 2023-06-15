
from typing import Tuple
from pandas import DataFrame, Series


PreprocessedData = Tuple[DataFrame, DataFrame, Series] # samples, samples_str, labels
SamplesAndLabels = Tuple[DataFrame, Series] # samples, labels
Samples = DataFrame # samples

def get_feature_column_names(
        preprocessed_data: PreprocessedData
    ) -> list[str]:
    """
    Get the names of the feature columns.
    """
    samples, _, _ = preprocessed_data
    return list(samples.columns)


def from_preprocessed_data_to_samples_and_labels(
        preprocessed_data: PreprocessedData
    ) -> SamplesAndLabels:
    """
    Get the samples and labels from the preprocessed data.
    """
    samples, _, labels = preprocessed_data
    return samples, labels


def from_preprocessed_data_to_samples(
        preprocessed_data: PreprocessedData
    ) -> Samples:
    """
    Get the samples from the preprocessed data.
    """
    samples, _, _ = preprocessed_data
    return samples