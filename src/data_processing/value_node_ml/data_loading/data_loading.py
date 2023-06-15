from concurrent.futures import ThreadPoolExecutor
import glob
from typing import Tuple, List
import os
import csv
from threading import Lock
import pandas as pd
import hashlib

from value_node_ml.data_loading.data_cleaning import clean
from value_node_ml.data_loading.data_types import PreprocessedData
from commons.utils.data_utils import count_positive_and_negative_labels
from commons.params.data_origin import DataOriginEnum
from value_node_ml.params.params import ProgramParams
from value_node_ml.params.dataset_loading_params import DatasetLoadingPossibilities


def __load_samples_and_labels_from_csv(
    csv_file_path: str,
    column_dtypes: dict[str, str] | str = "int32",
) -> PreprocessedData | None:
    # Load the data from the CSV file
    data = pd.read_csv(csv_file_path, dtype=column_dtypes)

    # Check if data is empty
    if data.empty:
        return None

    try:
        # Extract the labels from the last column
        labels = data.iloc[:, -1]

        # Extract the samples from the other columns
        samples = data.iloc[:, :-1]

    except Exception as e:
        raise type(e)(e.__str__() + f". Error loading data from {csv_file_path}")

    return samples, labels
        

def __load_samples_and_labels_from_all_csv_files(
        params: ProgramParams, csv_file_paths: list[str]
) -> PreprocessedData:
    """
    Load the samples and labels from all .csv files.
    """
    # stats
    list_of_empty_files = []

    all_samples_list: list[pd.DataFrame] = []
    all_labels_list: list[pd.Series] = []

    for i, csv_file_path in enumerate(csv_file_paths):

        # log the current file with respect to the total number of files
        params.COMMON_LOGGER.info(f'📋 [f: {i} / {len(csv_file_paths)}] Loading file {csv_file_path} ')

        res = __load_samples_and_labels_from_csv(csv_file_path)
        if res is None:
            list_of_empty_files.append(csv_file_path)
        else:
            samples, labels = res

            # log the shapes of the arrays
            params.COMMON_LOGGER.debug(f'shape of samples: {samples.shape}, shape of labels: {labels.shape}')

            all_samples_list.append(samples)
            all_labels_list.append(labels)

    params.COMMON_LOGGER.info(f'Number of empty files: {len(list_of_empty_files)}')

    # Concatenate DataFrames and labels Series
    all_samples = pd.concat(all_samples_list, ignore_index=True)
    all_labels = pd.concat(all_labels_list, ignore_index=True)

    return all_samples, all_labels


def __parallel_load_samples_and_labels_from_all_csv_files(
        params: ProgramParams, csv_file_paths: List[str]
) -> PreprocessedData:
    """
    Load the samples and labels from all .csv files.
    Load using multiple threads.
    """
    # stats
    list_of_empty_files = []

    all_samples_list: List[pd.DataFrame] = []
    all_labels_list: List[pd.Series] = []

    # determine header types
    first_file = csv_file_paths[0]
    header_types = __generate_dtype_dict(first_file)

    # Define a lock for thread safety
    concat_lock = Lock()

    def load_samples_and_labels_from_csv_and_concatenate(
        csv_file_path: str, 
        threadId: int, 
        nb_threads: int
    ) -> None:
        """
        Load the samples and labels from one .csv file.
        """
        params.RESULTS_LOGGER.info(f"📋 [{threadId}/{nb_threads}] Loading samples and labels from {csv_file_path}")

        res = __load_samples_and_labels_from_csv(csv_file_path, header_types)
        if res is None:
            list_of_empty_files.append(csv_file_path)
        else:
            samples, labels = res

            # Print the shapes of the arrays
            params.COMMON_LOGGER.debug(f'shape of samples: {samples.shape}, shape of labels: {labels.shape}')

            # Acquire the lock
            with concat_lock:
                all_samples_list.append(samples)
                all_labels_list.append(labels)
            # The lock is released after the 'with' statement

    # multi-threaded loading and generation of samples and labels
    with ThreadPoolExecutor(max_workers=min(params.MAX_ML_WORKERS, 6)) as executor:
        results = executor.map(
            load_samples_and_labels_from_csv_and_concatenate, 
            csv_file_paths,
            range(len(csv_file_paths)),
            [len(csv_file_paths)] * len(csv_file_paths)
        )
        for _ in results:
            pass

    params.COMMON_LOGGER.info(f'Number of empty files: {len(list_of_empty_files)}')

    # Concatenate DataFrames and labels Series
    all_samples = pd.concat(all_samples_list, ignore_index=True)
    all_labels = pd.concat(all_labels_list, ignore_index=True)

    return all_samples, all_labels


def get_all_filepath_per_type(dirpath: str) -> Tuple[list[str], list[str], list[str]]:
    """
    Determine the filepaths for all data .csv files inside the directory.
    Return the filepaths for the training, validation, and testing data.
    """
    extension = "csv"
    all_files = glob.glob(os.path.join(dirpath, "**", f"*.{extension}"), recursive=True)

    training_files = [file for file in all_files if DataOriginEnum.Training.value in file.lower()]
    validation_files = [file for file in all_files if DataOriginEnum.Validation.value in file.lower()]
    testing_files = [file for file in all_files if DataOriginEnum.Testing.value in file.lower()]
        
    return training_files, validation_files, testing_files


def log_positive_and_negative_labels(params: ProgramParams, labels: pd.Series, message: str = "") -> None:
    nb_positive_labels, nb_negative_labels = count_positive_and_negative_labels(labels)

    if message != "":
        params.RESULTS_LOGGER.info(message)

    params.RESULTS_LOGGER.info(f'Number of positive labels: {nb_positive_labels}')
    params.RESULTS_LOGGER.info(f'Number of negative labels: {nb_negative_labels}')


def __generate_dtype_dict(file_path: str, default_type: type = 'int32', special_cols_type: type = 'str', special_cols_keyword: str = 'path') -> dict:
    """
    Generate a dictionary for dtype parameter in pd.read_csv where any column containing 
    special_cols_keyword is of type special_cols_type, and all the others are of type default_type.

    :param file_path: path to the csv file
    :param default_type: default type for columns
    :param special_cols_type: type for special columns
    :param special_cols_keyword: keyword to identify special columns
    :return: dtype dict
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # get the first line, i.e., header

    # create dtype dict: column_name -> type
    dtype_dict = {col: special_cols_type if special_cols_keyword in col else default_type for col in header}

    return dtype_dict


def load(
        params: ProgramParams,
        data_origin: set[DataOriginEnum] | None = None
) -> PreprocessedData:
    """
    Load the samples and labels from all .csv files.
    Take into account the data origin: training, validation, testing.
    If data_origin is None, load all data.
    """

    # determine dataset directory
    match params.dataset:
        case DatasetLoadingPossibilities.LOAD_VALUE_NODE_DATASET:
            data_dir_path = params.CSV_DATASET_SAMPLES_AND_LABELS_DIR_PATH
        case DatasetLoadingPossibilities.LOAD_DATA_STRUCTURE_DATASET:
            data_dir_path = params.CSV_DATASET_DATA_STRUCTURE_DIR_PATH
        case _:
            raise ValueError(f"Invalid dataset value: {params.dataset}")

    
    # Get the filepaths for the training, validation, and testing data
    training_files, validation_files, testing_files = get_all_filepath_per_type(data_dir_path)

    files_to_load = []
    if data_origin is None:
        files_to_load = training_files + validation_files + testing_files
    else:
        for origin in data_origin:
            if origin == DataOriginEnum.Training:
                files_to_load += training_files
            elif origin == DataOriginEnum.Validation:
                files_to_load += validation_files
            elif origin == DataOriginEnum.Testing:
                files_to_load += testing_files
            else:
                raise ValueError(f"Unknown data origin: {origin}")


    #training_samples, training_labels = load_samples_and_labels_from_all_csv_files(params, training_files)
    samples, labels = __parallel_load_samples_and_labels_from_all_csv_files(params, files_to_load)

    samples, labels = clean(
        params,
        samples,
        labels,
    )

    log_positive_and_negative_labels(
        params, 
        labels, 
        "Loaded data: ({})".format(", ".join([origin.value for origin in data_origin])) if data_origin is not None else "No data"
    )

    samples, samples_str = separate_str_columns_from_features(samples)

    return samples, samples_str, labels


def compute_feature_vector_hash_id(
        row: pd.Series,
):
    """
    Compute the hash id of a feature vector.
    NOTE: the row must not contain string columns.
    """
    assert not any(isinstance(x, str) for x in row), "The row must not contain string columns."

    return hashlib.sha256(row.values.tobytes()).hexdigest()


def separate_str_columns_from_features(
    samples: pd.DataFrame
):
    """
    String columns are separated from the features and labels.
    These columns are used for identification purposes.
    They should NOT be used for training, fearure engineering, etc.
    """

    # Get the columns that contain strings
    # WARN: dtype 'str' is not supported by pandas, so we use 'object' instead
    str_columns = samples.select_dtypes(include='object').columns

    # Separate the string columns from the features and labels
    samples_str_columns = samples[str_columns]

    # We want to be able to find back the string columns of a given sample
    # for this, we need to hash every row of non-string columns,
    # and store the hash as a new column (hash_id) in the dataframe of string columns
    # so that, given a vector of features (with non-string columns), 
    # we can find back the string columns of that given feature vector

    # Get the non-string columns
    non_str_columns = samples.select_dtypes(exclude='object').columns

    # Compute a hash for each row of non-string columns
    samples_str_columns['hash_id'] = samples[non_str_columns].apply(
        lambda row: compute_feature_vector_hash_id(row), axis=1
    )

    # Drop the string columns from the original samples DataFrame
    samples = samples.drop(columns=str_columns)

    return samples, samples_str_columns
