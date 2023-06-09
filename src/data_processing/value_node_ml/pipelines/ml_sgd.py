from typing import Optional
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from commons.params.data_origin import DataOriginEnum

from value_node_ml.data_loading.data_types import SamplesAndLabelsGenerator, SamplesAndLabels, SamplesAndLabelsUnion, is_datagenerator, is_datatuple
from value_node_ml.data_loading.data_loading import consume_data_generator
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.pipelines.pipeline_utils import handle_data_origin_respecting_generator
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

    # Perform undersampling on the majority class
    #rus = RandomUnderSampler(random_state=42)
    #X_res, y_res = rus.fit_resample(X_train, y_train)
    X_res = X_train
    y_res = y_train

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

    


def __ml_sgd_pipeline_partial_fit(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabelsGenerator,
        samples_and_labels_test: Optional[SamplesAndLabelsGenerator],
    ) -> None:
    """
    Note that we need to consume the data generator for the testing part.
    """

    # Train a SGDClassifier
    clf = SGDClassifier(random_state=42)
    params.results_manager.set_result_for(
        PipelineNames.ML_SGD ,"model_name", "sgd"
    )
    X_test_all = pd.DataFrame()
    y_test_all = pd.Series()
    
    for samples_train, labels_train in samples_and_labels_train:

        if samples_and_labels_test is None:
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(samples_train, labels_train, test_size=0.2, random_state=42)
            X_test_all = pd.concat([X_test_all, X_test])
            y_test_all = pd.concat([y_test_all, y_test])
        else:
            X_train, y_train = samples_train, labels_train

        # Perform undersampling on the majority class
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)

        # Here classes=[0, 1] as we are assuming binary classification
        clf.partial_fit(X_res, y_res, classes=[0, 1])

    if samples_and_labels_test is not None:
        X_test_all, y_test_all = consume_data_generator(samples_and_labels_test)

    # Make predictions on the test set
    y_pred = clf.predict(X_test_all)

    # Compute metrics
    precision = precision_score(y_test_all, y_pred)
    recall = recall_score(y_test_all, y_pred)
    f1 = f1_score(y_test_all, y_pred)

    # Log the results
    params.RESULTS_LOGGER.info(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')



def ml_sgd_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
) -> None:
    samples_and_labels_train = handle_data_origin_respecting_generator(
        params.data_origins_training,
        origin_to_samples_and_labels
    )
    samples_and_labels_test = None
    if params.data_origins_testing is not None:
        samples_and_labels_test = handle_data_origin_respecting_generator(
            params.data_origins_testing,
            origin_to_samples_and_labels
        )

    if params.use_batch:
        __ml_sgd_pipeline_partial_fit(params, samples_and_labels_train, samples_and_labels_test)
    else:
        __ml_sgd_pipeline(params, samples_and_labels_train, samples_and_labels_test)

