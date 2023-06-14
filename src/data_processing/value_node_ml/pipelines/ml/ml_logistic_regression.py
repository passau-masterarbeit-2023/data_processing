from typing import Optional
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from commons.utils.ml_utils.ml_evaluate import evaluate
from commons.params.data_origin import DataOriginEnum
from value_node_ml.data_balancing.data_balancing import apply_balancing
from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import split_dataset_if_needed, split_samples_and_labels
from value_node_ml.pipelines.feature_engineering.univariate_feature_selection import __compute_distance_f_test_p_val
from value_node_ml.params.params import ProgramParams

def __ml_logistic_regression_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:

    X_train, X_test, y_train, y_test = split_dataset_if_needed(
        samples_and_labels_train, samples_and_labels_test
    )
    
    # balance data
    X_train, y_train = apply_balancing(params, X_train, y_train, PipelineNames.ML_LOGISTIC_REG)

    # Train classifier
    clf = LogisticRegression(n_jobs = params.MAX_ML_WORKERS)
    clf.fit(X_train, y_train)
    params.ml_results_manager.set_result_for(
        PipelineNames.ML_LOGISTIC_REG ,"model_name", "LogisticRegression"
    )

    # Evaluate model
    evaluate(
        clf,
        X_test,
        y_test,
        params.RESULTS_LOGGER,
        params.ml_results_manager.get_result_writer_for(PipelineNames.ML_LOGISTIC_REG),
    )


def ml_logistic_regression_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
    ) -> None:
    """
    Pipeline for training a logistic regression model.
    """

    samples_and_labels_train, samples_and_labels_test, = split_samples_and_labels(
        params, origin_to_samples_and_labels
    )
    
    # launch the pipeline
    __ml_logistic_regression_pipeline(params, samples_and_labels_train, samples_and_labels_test)
