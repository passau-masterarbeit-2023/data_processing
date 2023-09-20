
from typing import Optional, Tuple
from pandas import DataFrame, Series


PreprocessedData = Tuple[DataFrame, DataFrame, Series] # samples, samples_str, labels
SamplesAndLabels = Tuple[DataFrame, Series] # samples, labels
SamplesAndSamplesStr = Tuple[DataFrame, DataFrame] # samples, samples_str
Samples = DataFrame # samples

def get_feature_column_names(
    preprocessed_data: Optional[PreprocessedData]
) -> list[str]:
    """
    Get the names of the feature columns.
    """
    if preprocessed_data is None:
        return None

    samples, _, _ = preprocessed_data
    return list(samples.columns)


def from_preprocessed_data_to_samples_and_labels(
    preprocessed_data: Optional[PreprocessedData]
):
    """
    Get the samples and labels from the preprocessed data.
    """
    if preprocessed_data is None:
        return None

    samples, _, labels = preprocessed_data
    return samples, labels


def from_preprocessed_data_to_samples(
    preprocessed_data: Optional[PreprocessedData]
):
    """
    Get the samples from the preprocessed data.
    """
    if preprocessed_data is None:
        return None

    samples, _, _ = preprocessed_data
    return samples


def from_preprocessed_data_to_samples_and_sample_str(
    preprocessed_data: Optional[PreprocessedData]
):
    """
    Keep only the samples and labels from the preprocessed data.
    """
    if preprocessed_data is None:
        return None

    samples, samples_str, _ = preprocessed_data
    return samples, samples_str


