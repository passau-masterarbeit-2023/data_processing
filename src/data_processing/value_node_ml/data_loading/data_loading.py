from concurrent.futures import ThreadPoolExecutor
import glob
from typing import Tuple, List
import os
from threading import Lock
import pandas as pd
from typing import Generator, Tuple

from value_node_ml.data_loading.data_cleaning import clean
from value_node_ml.data_loading.data_types import SamplesAndLabelsGenerator, SamplesAndLabels, SamplesAndLabelsUnion
from commons.utils.data_utils import count_positive_and_negative_labels
from commons.params.data_origin import DataOriginEnum
from commons.utils.utils import time_measure_result
from value_node_ml.params.params import ProgramParams


def load_samples_and_labels_from_csv(csv_file_path: str) -> SamplesAndLabels | None:
    # Load the data from the CSV file
    data = pd.read_csv(csv_file_path, dtype='int32')

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


def load_samples_and_labels_from_all_csv_files_in_batches(
        params: ProgramParams, 
        csv_file_paths: list[str],
) -> SamplesAndLabelsGenerator:
    """
    Generator function to yield batches of samples and labels from a CSV file.
    One batch is one file.
    """
    # stats
    list_of_empty_files = []

    for i, csv_file_path in enumerate(csv_file_paths):

        # log the current file with respect to the total number of files
        params.COMMON_LOGGER.info(f'📋 [f: {i} / {len(csv_file_paths)}] Loading file {csv_file_path} ')

        res = load_samples_and_labels_from_csv(csv_file_path)
        if res is None:
            list_of_empty_files.append(csv_file_path)
        else:
            samples, labels = res

            # log the shapes of the arrays
            params.COMMON_LOGGER.debug(f'shape of samples: {samples.shape}, shape of labels: {labels.shape}')

            samples, labels = clean(
                params,
                samples,
                labels,
            )

            yield samples, labels

    params.COMMON_LOGGER.info(f'Number of empty files: {len(list_of_empty_files)}')


def consume_data_generator(data_generator: SamplesAndLabelsGenerator) -> SamplesAndLabels:
    """
    Consume a data generator.
    """
    all_samples_list: list[pd.DataFrame] = []
    all_labels_list: list[pd.Series] = []

    for samples, labels in data_generator:
        all_samples_list.append(samples)
        all_labels_list.append(labels)
    
    # Concatenate DataFrames and labels Series
    all_samples = pd.concat(all_samples_list, ignore_index=True)
    all_labels = pd.concat(all_labels_list, ignore_index=True)

    return all_samples, all_labels
        

def load_samples_and_labels_from_all_csv_files(
        params: ProgramParams, csv_file_paths: list[str]
) -> SamplesAndLabels:
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

        res = load_samples_and_labels_from_csv(csv_file_path)
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


def parallel_load_samples_and_labels_from_all_csv_files(
        params: ProgramParams, csv_file_paths: List[str]
) -> SamplesAndLabels:
    """
    Load the samples and labels from all .csv files.
    Load using multiple threads.
    """
    # stats
    list_of_empty_files = []

    all_samples_list: List[pd.DataFrame] = []
    all_labels_list: List[pd.Series] = []

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

        res = load_samples_and_labels_from_csv(csv_file_path)
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


def load(
        params: ProgramParams, 
        data_dir_path: str, 
        data_origin: set[DataOriginEnum] | None = None
) -> SamplesAndLabelsUnion:
    """
    Load the samples and labels from all .csv files.
    Take into account the data origin: training, validation, testing.
    If data_origin is None, load all data.
    """
    
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

    # Load the training data
    if not params.use_batch:
        #training_samples, training_labels = load_samples_and_labels_from_all_csv_files(params, training_files)
        samples, labels = parallel_load_samples_and_labels_from_all_csv_files(params, files_to_load)

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

        return samples, labels
    else:
        return load_samples_and_labels_from_all_csv_files_in_batches(params, files_to_load)

    