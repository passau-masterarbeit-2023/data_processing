from typing import Optional
from sklearn.linear_model import SGDClassifier
from commons.params.data_origin import DataOriginEnum

from value_node_ml.data_balancing.data_balancing import apply_balancing
from value_node_ml.data_loading.data_types import PreprocessedData, from_preprocessed_data_to_samples_and_labels
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import split_dataset_if_needed, split_preprocessed_data_by_origin
from value_node_ml.params.params import ProgramParams
from commons.utils.ml_utils.ml_evaluate import evaluate

def __ml_sgd_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: PreprocessedData,
        samples_and_labels_test: Optional[PreprocessedData],
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
        origin_to_preprocessed_data: dict[DataOriginEnum, PreprocessedData]
) -> None:

    preprocessed_data_train, preprocessed_data_test = split_preprocessed_data_by_origin(
        params, origin_to_preprocessed_data
    )

    print("type(preprocessed_data_train)", type(preprocessed_data_train))
    print(len(preprocessed_data_train))
    for item in preprocessed_data_train:
        print(type(item))
        print(len(item))

    samples_and_labels_train = from_preprocessed_data_to_samples_and_labels(preprocessed_data_train)
    samples_and_labels_test = from_preprocessed_data_to_samples_and_labels(preprocessed_data_test)
    
    # launch the pipeline
    __ml_sgd_pipeline(params, samples_and_labels_train, samples_and_labels_test)