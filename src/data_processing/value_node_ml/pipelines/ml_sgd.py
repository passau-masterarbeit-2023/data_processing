from typing import Optional
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from commons.params.data_origin import DataOriginEnum

from value_node_ml.data_balancing.data_balancing import apply_balancing
from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import split_samples_and_labels
from value_node_ml.params.params import ProgramParams

def __ml_sgd_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:
    """
    Pipeline for SGDClassifier with undersampling.
    """
    if samples_and_labels_test is None:
        # Split data into training and test sets
        samples, labels = samples_and_labels_train
        X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = samples_and_labels_train
        X_test, y_test = samples_and_labels_test

    # data balancing
    X_res, y_res = apply_balancing(params, X_train, y_train, PipelineNames.ML_SGD)

    # Train a SGDClassifier
    print(params.MAX_ML_WORKERS)
    clf = SGDClassifier(random_state=42, n_jobs = params.MAX_ML_WORKERS)
    params.results_manager.set_result_for(
        PipelineNames.ML_SGD ,"model_name", "sgd"
    )

    # Train classifier
    clf.fit(X_res, y_res)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Compute metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log the results
    params.RESULTS_LOGGER.info(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')


def ml_sgd_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
) -> None:
    
    samples_and_labels_train, samples_and_labels_test, = split_samples_and_labels(
        params, origin_to_samples_and_labels
    )

    __ml_sgd_pipeline(params, samples_and_labels_train, samples_and_labels_test)

