from typing import Optional
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from value_node_ml.data_loading.data_types import SamplesAndLabelsGenerator, SamplesAndLabels, SamplesAndLabelsUnion, is_datagenerator, is_datatuple
from value_node_ml.data_loading.data_loading import consume_data_generator
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.params.data_origin import DataOriginEnum
from value_node_ml.pipelines.pipeline_utils import handle_data_origin_consume_generator
from value_node_ml.params.params import ProgramParams


def __compute_distance_f_test_p_val(f_values: np.ndarray, p_values: np.ndarray) -> np.ndarray:
    """
    Compute the distance between the F-test value and the p-value.
    """
    # Normalize F-test values and p-values to the range [0, 1]
    f_values_norm = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
    p_values_norm = (p_values - np.min(p_values)) / (np.max(p_values) - np.min(p_values))

    # Compute the combined importance value for each feature
    combined_values = f_values_norm - p_values_norm

    # Sort the features based on the combined importance values (descending)
    sorted_indices = np.argsort(-combined_values)

    return sorted_indices


def __univariate_feature_selection_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:
    """
    Pipeline for univariate feature selection.
    """

    params.results_manager.set_result_for(
        PipelineNames.UNIVARIATE_FS ,"model_name", "univariate_feature_selection"
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
    sorted_indices = __compute_distance_f_test_p_val(f_values, p_values)

    # Map the sorted indices to the names of the columns
    sorted_column_names = [column_names[i] for i in sorted_indices]

    # Print the sorted column names
    params.RESULTS_LOGGER.info(f"Column names sorted by importance: [{', '.join(sorted_column_names)}]")

    # TODO: evaluate the performance of the classifier with the selected features
    #selector.fit(training_samples, training_labels)
    #X_new = selector.transform(training_samples)


def univariate_feature_selection_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
    ) -> None:
    """
    Pipeline for feature selection.
    """

    samples_and_labels_train: SamplesAndLabels 
    samples_and_labels_test: Optional[SamplesAndLabels] = None

    samples_and_labels_train = handle_data_origin_consume_generator(
        params.data_origins_training,
        origin_to_samples_and_labels
    )
    if params.data_origins_testing is not None:
        samples_and_labels_test = handle_data_origin_consume_generator(
            params.data_origins_testing,
            origin_to_samples_and_labels
        )
    
    # launch the pipeline
    __univariate_feature_selection_pipeline(params, samples_and_labels_train, samples_and_labels_test)

