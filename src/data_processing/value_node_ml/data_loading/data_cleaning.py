from value_node_ml.params.params import ProgramParams
from typing import Tuple
import pandas as pd



def __remove_unecessary_columns(params: ProgramParams, samples: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    When the user provides a list of columns to keep, remove the unecessary columns.
    Otherwise, keep all columns (do nothing).

    NOTE: This is useful for testing the pipelines on a subset of the data, 
    especially as a result of (FE) feature engineering (feature selection, etc.)
    """
    if params.columns_to_keep is None:
        return samples, labels
    
    # Log the removed columns
    removed_columns = set(samples.columns) - params.columns_to_keep
    params.RESULTS_LOGGER.info(f'Removing {len(removed_columns)} columns: {list(removed_columns)}')

    # Remove the columns
    samples = samples[list(params.columns_to_keep)]
    return samples, labels


def clean(params: ProgramParams, samples: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean data.
    1. Remove columns that are composed off only one value
    2. Remove unecessary columns (when usefull ones are provided)
    """
    # Find the indices of columns with only one unique value
    unique_value_columns = samples.nunique() == 1

    # Log the removed columns
    removed_columns = unique_value_columns[unique_value_columns].index
    params.RESULTS_LOGGER.info(f'Removing {len(removed_columns)} columns with only one unique value: {list(removed_columns)}')

    # Remove the columns with only one unique value from the samples
    samples = samples.loc[:, ~unique_value_columns]

    # step 2
    samples, labels = __remove_unecessary_columns(params, samples, labels)

    return samples, labels

