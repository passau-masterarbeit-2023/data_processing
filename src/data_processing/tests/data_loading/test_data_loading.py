
from processing_pipelines.data_loading.data_loading import __load_samples_and_labels_from_csv
from commons.utils.data_utils import count_positive_and_negative_labels
from tests.tests import init_test


def test_load_samples_and_labels_from_csv():
    testParams = init_test()

    samples, labels = __load_samples_and_labels_from_csv(
        testParams.TEST_CSV_TRAINING_1
    )

    positives, negative = count_positive_and_negative_labels(labels)
    print(f'Number of positive labels: {positives}')
    print(f'Number of negative labels: {negative}')

    # results from data/test/samples_and_labels/Training__chunck_idx-0_samples.csv, manually counted
    assert positives == 600
    assert negative == 873423 - 1 - 600 # 1 for the header, 600 for the positive labels

    # number of features, number of samples
    print(f'Number of samples: {samples.shape[0]}')
    print(f'Number of features: {samples.shape[1]}')

    assert samples.shape[0] == 873423 - 1 # 1 for the header
    assert samples.shape[1] == 18