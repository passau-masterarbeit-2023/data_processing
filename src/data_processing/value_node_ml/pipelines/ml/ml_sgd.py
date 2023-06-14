from typing import Optional
from sklearn.linear_model import SGDClassifier
from commons.params.data_origin import DataOriginEnum

from value_node_ml.data_balancing.data_balancing import apply_balancing
from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import split_dataset_if_needed, split_samples_and_labels
from value_node_ml.params.params import ProgramParams
from commons.utils.ml_utils.ml_evaluate import evaluate

def __ml_sgd_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:
    """
    Pipeline for SGDClassifier with undersampling.
    """
    X_train, X_test, y_train, y_test = split_dataset_if_needed(
        samples_and_labels_train, samples_and_labels_test
    )

    # data balancing
    X_res, y_res = apply_balancing(params, X_train, y_train, PipelineNames.ML_SGD)

    # Train a SGDClassifier
    clf = SGDClassifier(random_state=42, n_jobs = params.MAX_ML_WORKERS)
    params.ml_results_manager.set_result_for(
        PipelineNames.ML_SGD ,"model_name", "sgd"
    )

    # Train classifier
    clf.fit(X_res, y_res)

    # Evaluate model
    evaluate(
        clf,
        X_test,
        y_test,
        params.RESULTS_LOGGER,
        params.ml_results_manager.get_result_writer_for(PipelineNames.ML_SGD),
    )


def ml_sgd_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
) -> None:
    
    samples_and_labels_train, samples_and_labels_test, = split_samples_and_labels(
        params, origin_to_samples_and_labels
    )

    __ml_sgd_pipeline(params, samples_and_labels_train, samples_and_labels_test)

