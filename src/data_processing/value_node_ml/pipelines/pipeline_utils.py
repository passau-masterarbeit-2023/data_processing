
import pandas as pd
from value_node_ml.data_loading.data_loading import consume_data_generator
from value_node_ml.data_loading.data_types import SamplesAndLabelsGenerator, SamplesAndLabels, SamplesAndLabelsUnion, is_datagenerator, is_datatuple
from value_node_ml.params.data_origin import DataOriginEnum


def handle_data_origin_consume_generator(
    data_origins: set[DataOriginEnum],
    origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
):
    """
    Helper function for handling data origins, for classifiers that do not use data generators.
    """
    __samples: list[pd.DataFrame] = []
    __labels: list[pd.Series] = []
    for origin in data_origins:
        samples_and_labels = origin_to_samples_and_labels[origin]

        if is_datatuple(samples_and_labels):
            samples, labels = samples_and_labels
            __labels += [labels]
            __samples += [samples]
        elif is_datagenerator(samples_and_labels):
            # consume the data generator
            samples, labels = consume_data_generator(samples_and_labels)
            __labels += [labels]
            __samples += [samples]
        else:
            raise TypeError(f"Invalid type for samples_and_labels: {type(samples_and_labels)}")
    samples_and_labels = (pd.concat(__samples), pd.concat(__labels))
    return samples_and_labels


def handle_data_origin_respecting_generator(
    data_origins: set[DataOriginEnum],
    origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
) -> SamplesAndLabelsUnion:
    """
    Helper function for handling data origins, for classifiers that potentially use data generators.
    """

    # beware interperter on "yield from". Need to take it to its own function to avoid skipping control flow
    def concat_generators(
            data_origins: set[DataOriginEnum],
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabelsUnion]
    ):
        for data_origin in data_origins:
            # NOTE: yield from allows to yield from a generator from within a generator
            # ameno: http://simeonvisser.com/posts/python-3-using-yield-from-in-generators-part-1.html
            yield from origin_to_samples_and_labels[data_origin]


    first_samples_and_labels = next(iter(origin_to_samples_and_labels.values())) # get first value, without iterating
    if is_datagenerator(first_samples_and_labels):
        return concat_generators(data_origins, origin_to_samples_and_labels)
    elif is_datatuple(first_samples_and_labels):
        vals = handle_data_origin_consume_generator(data_origins, origin_to_samples_and_labels)
        return vals
    else:
        raise TypeError(f"Invalid type for samples_and_labels: {type(first_samples_and_labels)}")
            