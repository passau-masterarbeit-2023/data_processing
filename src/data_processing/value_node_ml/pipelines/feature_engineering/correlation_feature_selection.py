from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd

from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.params.params import ProgramParams
from commons.params.data_origin import DataOriginEnum
from value_node_ml.pipelines.pipeline_utils import split_samples_and_labels
from commons.utils.utils import DATETIME_FORMAT

def __correlation_feature_selection(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        correlation_algorithm: str,
        start_timestamp: str,
    ) -> list[str]:
    """
    Pipeline for feature engineering correlation measurement.
    return: best columns names
    """
    params.COMMON_LOGGER.info(f"Computing correlation (algorithm: {correlation_algorithm})...")
    
    # Extract samples from training data
    samples, _ = samples_and_labels_train

    # scale the samples to avoid overflows (returns a numpy array)
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(samples)

    # Convert scaled_samples back to DataFrame
    scaled_samples_df = pd.DataFrame(scaled_samples, columns=samples.columns)

    # Calculate Pearson correlation matrix
    corr_matrix = scaled_samples_df.corr(correlation_algorithm)

    # Print the correlation matrix
    params.COMMON_LOGGER.info(f"Correlation matrix (algorithm: {correlation_algorithm}): \n" + str(corr_matrix))

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True, cmap='coolwarm')
    plt.title(f"Feature Correlation Matrix (algorithm: {correlation_algorithm})")
    corr_matrix_save_path: str = (
        params.FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH + 
        "correlation_matrix_" + correlation_algorithm + "_" +
        start_timestamp +
        ".png"
    )
    plt.savefig(corr_matrix_save_path)
    plt.close()

    # keep best columns
    # Calculate the sum of correlations for each column
    corr_sums = corr_matrix.abs().sum()
    
    # Find the names of the columns that have the smallest sums
    # NOTE: We drop the 1 correlation of the column with itself by substracting 1 to the sums
    corr_sums -= 1
    best_columns_names = corr_sums.nsmallest(params.FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS).index.tolist()
    
    params.COMMON_LOGGER.info(f"Keeping columns: {best_columns_names}")

    assert len(best_columns_names) == params.FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS
    assert (type(best_columns_names) == list) and (type(best_columns_names[0]) == str)

    # return the best columns names
    return best_columns_names


def __feature_engineering_correlation_selection(
        params: ProgramParams, 
        samples_and_labels_train: SamplesAndLabels,
        samples_and_labels_test: Optional[SamplesAndLabels],
    ) -> None:
    """
    Pipeline for feature engineering correlation measurement.
    NOTE: Correlation algorithms: pearson, kendall, spearman
    """
    correlation_algotithms = ["pearson", "kendall", "spearman"]

    start_timestamp = datetime.now().strftime(DATETIME_FORMAT)
    for correlation_algorithm in correlation_algotithms:
        __correlation_feature_selection(
            params, samples_and_labels_train, correlation_algorithm, start_timestamp
        )
        

def feature_engineering_correlation_measurement_pipeline(
        params: ProgramParams, 
        origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels]
    ) -> None:
    """
    Pipeline for feature engineering correlation measurement.
    """

    samples_and_labels_train, samples_and_labels_test, = split_samples_and_labels(
        params, origin_to_samples_and_labels
    )
    
    # launch the pipeline
    __feature_engineering_correlation_selection(params, samples_and_labels_train, samples_and_labels_test)