from enum import Enum

from commons.utils.enum_utils import print_enum_values, convert_str_arg_to_enum_member

class DatasetLoadingPossibilities(Enum):
    """
    Load possibilities.
    """
    LOAD_VALUE_NODE_DATASET = "load_value_node_dataset"
    LOAD_DATA_STRUCTURE_DATASET = "load_data_structure_dataset"


def print_load_possibilities():
    """
    Print the possible values of the LoadPossibilities.
    """
    print_enum_values(DatasetLoadingPossibilities)


def convert_str_arg_to_dataset(arg: str):
    """
    Convert a string argument to a LoadPossibilities.
    """
    return convert_str_arg_to_enum_member(arg, DatasetLoadingPossibilities)

def check_value_node_dataset_params(dataset: DatasetLoadingPossibilities):
    """
    Check that the dataset is a value node dataset.
    """
    if dataset != DatasetLoadingPossibilities.LOAD_VALUE_NODE_DATASET:
        print(f"ERROR: Invalid dataset for the specified pipelines: {dataset}")
        print(f"ERROR: The dataset should be: {DatasetLoadingPossibilities.LOAD_VALUE_NODE_DATASET}")
        exit(1)

def check_data_structure_dataset_params(dataset: DatasetLoadingPossibilities):
    """
    Check that the dataset is a data structure dataset.
    """
    if dataset != DatasetLoadingPossibilities.LOAD_DATA_STRUCTURE_DATASET:
        print(f"ERROR: Invalid dataset for the specified pipelines: {dataset}")
        print(f"ERROR: The dataset should be: {DatasetLoadingPossibilities.LOAD_DATA_STRUCTURE_DATASET}")
        exit(1)