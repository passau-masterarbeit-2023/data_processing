from value_node_ml.data_loading.data_loading import load_samples_and_labels_from_csv
from value_node_ml.pipelines.check import __check_samples_and_labels
from tests.tests import init_test

def test_check_samples_and_labels():
    testParams = init_test()

    samples, labels = load_samples_and_labels_from_csv(
        testParams.TEST_CSV_TRAINING_1
    )

    __check_samples_and_labels(testParams, samples, labels)
    