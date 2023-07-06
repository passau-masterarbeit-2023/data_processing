from typing import Optional
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from processing_pipelines.data_loading.data_types import PreprocessedData, from_preprocessed_data_to_samples_and_labels
from processing_pipelines.params.pipeline_params import PipelineNames
from commons.params.data_origin import DataOriginEnum
from processing_pipelines.pipelines.pipeline_utils import split_preprocessed_data_by_origin
from processing_pipelines.params.params import ProgramParams


def __compute_distance_f_test_p_val(f_values: np.ndarray, p_values: np.ndarray):
    """
    Compute the distance between the F-test value and the p-value.
    """
    # Normalize F-test values and p-values to the range [0, 1]
    f_values_norm: np.ndarray = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
    p_values_norm: np.ndarray = (p_values - np.min(p_values)) / (np.max(p_values) - np.min(p_values))

    # Compute the combined importance value for each feature
    combined_values: np.ndarray = f_values_norm - p_values_norm

    # Sort the features based on the combined importance values (descending)
    sorted_indices = np.argsort(-combined_values) # descending order

    return sorted_indices, -np.sort(-combined_values) # descending order


def __univariate_feature_selection_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: PreprocessedData,
    ) -> None:
    """
    Pipeline for univariate feature selection.
    """
    params.fe_results_manager.set_result_for(
        PipelineNames.FE_UNIVARIATE ,
        "feature_engineering_algorithm", 
        "select_k_best"
    )

    # Feature selection on training set (only)
    samples, labels = samples_and_labels_train

    # feature selection
    selector = SelectKBest(f_classif, k=10)
    f_values, p_values = selector.score_func(samples, labels)
    column_names = samples.columns.tolist()

    for name, f_value, p_value in zip(column_names, f_values, p_values):
        params.COMMON_LOGGER.info(f'Column: {name}, F-value: {f_value}, P-value: {p_value}')


    # Sort the features based on F-statistic values (descending) and p-values (ascending)
    sorted_indices, descending_combined_values = __compute_distance_f_test_p_val(f_values, p_values)

    # Map the sorted indices to the names of the columns
    sorted_column_names = [column_names[i] for i in sorted_indices]

    # Print the sorted column names
    params.RESULTS_LOGGER.info(f"Column names sorted by (F_val, P_val) importance: [{', '.join(sorted_column_names)}]")

    # keep results
    params.fe_results_manager.set_result_for(
        PipelineNames.FE_UNIVARIATE,
        "descending_best_column_names",
        " ".join(
            [str(name) for name in sorted_column_names]
        )
    )
    params.fe_results_manager.set_result_for(
        PipelineNames.FE_UNIVARIATE,
        "descending_best_column_values",
        " ".join(
            [str(value) for value in descending_combined_values]
        )
    )


def univariate_feature_selection_pipeline(
        params: ProgramParams, 
        origin_to_preprocessed_data: dict[DataOriginEnum, PreprocessedData]
    ) -> None:
    """
    Pipeline for feature selection.
    """

    preprocessed_data_train, _ = split_preprocessed_data_by_origin(
        params, origin_to_preprocessed_data
    )

    samples_and_labels_train = from_preprocessed_data_to_samples_and_labels(preprocessed_data_train)
    
    # launch the pipeline
    __univariate_feature_selection_pipeline(
        params, 
        samples_and_labels_train
    )

