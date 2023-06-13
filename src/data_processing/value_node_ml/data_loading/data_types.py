
from typing import Tuple
from pandas import DataFrame, Series


SamplesAndLabels = Tuple[DataFrame, Series]

def get_feature_column_names(samples_and_labels: SamplesAndLabels) -> list[str]:
    """
    Get the names of the feature columns.
    """
    samples, _ = samples_and_labels
    return list(samples.columns)