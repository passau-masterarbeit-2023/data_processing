from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd

from processing_pipelines.params.pipeline_params import PipelineNames
from processing_pipelines.data_loading.data_types import PreprocessedData, from_preprocessed_data_to_samples_and_labels
from processing_pipelines.params.params import ProgramParams
from commons.params.data_origin import DataOriginEnum
from processing_pipelines.pipelines.pipeline_utils import split_preprocessed_data_by_origin
from commons.utils.utils import DATETIME_FORMAT

def __correlation_feature_selection(
        params: ProgramParams, 
        samples_and_labels_train: PreprocessedData,
        pipeline_name: PipelineNames,

    ) -> list[str]:
    """
    Pipeline for feature engineering correlation measurement.
    return: best columns names
    """
    # select algorithm
    correlation_algorithms = {
        PipelineNames.FE_CORR_PEARSON: "pearson",
        PipelineNames.FE_CORR_KENDALL: "kendall",
        PipelineNames.FE_CORR_SPEARMAN: "spearman",
    }
    correlation_algorithm = correlation_algorithms[pipeline_name]

    # log and results
    params.COMMON_LOGGER.info(f"Computing correlation (algorithm: {correlation_algorithm})...")
    params.fe_results_manager.set_result_for(
        pipeline_name,
        "feature_engineering_algorithm",
        correlation_algorithm
    )

    # Extract samples from training data
    samples, _ = samples_and_labels_train

    # scale the samples to avoid overflows (returns a numpy array)
    scaler = StandardScaler()
    scaled_samples = scaler.fit_transform(samples)

    # Convert scaled_samples back to DataFrame
    scaled_samples_df = pd.DataFrame(scaled_samples, columns=samples.columns)

    # Calculate correlation matrix
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
        datetime.now().strftime(DATETIME_FORMAT) +
        ".png"
    )
    plt.savefig(corr_matrix_save_path)
    plt.close()

    # keep best columns
    # Calculate the sum of correlations for each column
    corr_sums = corr_matrix.abs().sum()

    # keep results
    sorted_corr_sums = corr_sums.sort_values(ascending=False)
    params.fe_results_manager.set_result_for(
        pipeline_name,
        "descending_best_column_names",
        " ".join(
            sorted_corr_sums.index.tolist()
        )
    )
    params.fe_results_manager.set_result_for(
        pipeline_name,
        "descending_best_column_values",
        " ".join(
            str(sorted_corr_sums.values.tolist())
        )
    )
    
    # Find the names of the columns that have the smallest sums
    # NOTE: We drop the 1 correlation of the column with itself by substracting 1 to the sums
    corr_sums -= 1
    best_columns_names = corr_sums.nsmallest(params.FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS).index.tolist()
    
    params.COMMON_LOGGER.info(f"Keeping columns: {best_columns_names}")

    assert len(best_columns_names) == params.FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS, "The number of best columns is not correct, it should be equal to FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS. Maybe there are not enough columns in the dataset."
    assert (type(best_columns_names) == list) and (type(best_columns_names[0]) == str)

    # return the best columns names
    return best_columns_names


def __feature_engineering_correlation_measurement_pipeline(
        params: ProgramParams, 
        origin_to_preprocessed_data: dict[DataOriginEnum, PreprocessedData],
        pipeline_name: PipelineNames,
    ) -> None:
    """
    Pipeline for feature engineering correlation measurement.
    """

    preprocessed_data_train, _ = split_preprocessed_data_by_origin(
        params, origin_to_preprocessed_data
    )

    samples_and_labels_train = from_preprocessed_data_to_samples_and_labels(preprocessed_data_train)
    
    # launch the pipeline
    __correlation_feature_selection(
        params, 
        samples_and_labels_train,
        pipeline_name
    )


# pipelines functions
def feature_engineering_correlation_measurement_pipeline_pearson(
    params: ProgramParams, 
    origin_to_samples_and_labels: dict[DataOriginEnum, PreprocessedData]
) -> None:
    __feature_engineering_correlation_measurement_pipeline(
        params, origin_to_samples_and_labels, PipelineNames.FE_CORR_PEARSON
    )

def feature_engineering_correlation_measurement_pipeline_kendall(
    params: ProgramParams, 
    origin_to_samples_and_labels: dict[DataOriginEnum, PreprocessedData]
) -> None:
    __feature_engineering_correlation_measurement_pipeline(
        params, origin_to_samples_and_labels, PipelineNames.FE_CORR_KENDALL
    )

def feature_engineering_correlation_measurement_pipeline_spearman(
    params: ProgramParams, 
    origin_to_samples_and_labels: dict[DataOriginEnum, PreprocessedData]
) -> None:
    __feature_engineering_correlation_measurement_pipeline(
        params, origin_to_samples_and_labels, PipelineNames.FE_CORR_SPEARMAN
    )

