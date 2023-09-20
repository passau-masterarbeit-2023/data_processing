from processing_pipelines.params.pipeline_params import PipelineNames
from processing_pipelines.pipelines.classification.ml_sgd import ml_sgd_pipeline
from processing_pipelines.pipelines.feature_engineering.univariate_feature_selection import univariate_feature_selection_pipeline
from processing_pipelines.pipelines.classification.ml_logistic_regression import ml_logistic_regression_pipeline
from processing_pipelines.pipelines.check import check
from processing_pipelines.pipelines.classification.ml_random_forest import ml_random_forest_pipeline
from processing_pipelines.pipelines.feature_engineering.correlation_feature_selection import *
from processing_pipelines.pipelines.data_structure_clustering.density_clustering import density_clustering_pipeline

PIPELINE_NAME_TO_FUNCTION = {
    PipelineNames.CHECK_VN : check,
    PipelineNames.FE_UNIVARIATE : univariate_feature_selection_pipeline,
    PipelineNames.ML_LOGISTIC_REG : ml_logistic_regression_pipeline,
    PipelineNames.ML_RANDOM_FOREST : ml_random_forest_pipeline,
    PipelineNames.ML_SGD : ml_sgd_pipeline,
    PipelineNames.FE_CORR_PEARSON: feature_engineering_correlation_measurement_pipeline_pearson,
    PipelineNames.FE_CORR_KENDALL: feature_engineering_correlation_measurement_pipeline_kendall,
    PipelineNames.FE_CORR_SPEARMAN: feature_engineering_correlation_measurement_pipeline_spearman,
    PipelineNames.DS_DENSITY_CLUSTERING: density_clustering_pipeline,
}
