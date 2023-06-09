from typing import Optional
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from commons.utils.ml_utils.ml_evaluate import evaluate
from value_node_ml.data_loading.data_types import SamplesAndLabels, SamplesAndLabelsUnion
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import handle_data_origin_consume_generator
from value_node_ml.params.data_origin import DataOriginEnum
from value_node_ml.pipelines.univariate_feature_selection import __compute_distance_f_test_p_val
from value_node_ml.params.params import ProgramParams

def __ml_logistic_regression_pipeline(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:

    if samples_and_labels_test is None:
        # Split data into training and test sets
        __samples, __labels = samples_and_labels_train
        X_train, X_test, y_train, y_test = train_test_split(__samples, __labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = samples_and_labels_train
        X_test, y_test = samples_and_labels_test

    # Feature selection
    selector = SelectKBest(f_classif, k=10)
    X_train_transformed = selector.fit_transform(X_train, y_train)

    # log selected features
    column_names_after_selection = X_train.columns[selector.get_support()].tolist()
    params.RESULTS_LOGGER.info(f'Selected features: {column_names_after_selection}')

    f_values, p_values = selector.score_func(X_train, y_train)
    column_names = X_train.columns.tolist()

    # Train classifier
    clf = LogisticRegression(n_jobs = params.MAX_ML_WORKERS)
    clf.fit(X_train_transformed, y_train)
    params.results_manager.set_result_for(
        PipelineNames.ML_LOGISTIC_REG ,"model_name", "LogisticRegression"
    )

    # Apply feature selection to test set
    X_test_transformed = selector.transform(X_test)

    # Evaluate model
    evaluate(
        clf,
        X_test_transformed,
        y_test,
        params.RESULTS_LOGGER,
        params.results_manager.get_result_writer_for(PipelineNames.ML_LOGISTIC_REG),
    )

    # Return feature importance
    f_values, p_values = selector.score_func(X_train, y_train)
    sorted_indices = __compute_distance_f_test_p_val(f_values, p_values)
    sorted_column_names = [column_names[i] for i in sorted_indices]
    params.RESULTS_LOGGER.info(f"Column names sorted by importance: [{', '.join(sorted_column_names)}]")




def ml_logistic_regression_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
    ) -> None:
    """
    Pipeline for training a logistic regression model.
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
    __ml_logistic_regression_pipeline(params, samples_and_labels_train, samples_and_labels_test)
