from enum import Enum

from commons.utils.enum_utils import print_enum_values, convert_str_arg_to_enum_member


class PipelineNames(Enum):
    """
    Pipeline names.
    """
    ##########common pipelines ##########
    # feature engineering
    FE_UNIVARIATE = "fe_univariate"
    FE_CORR_PEARSON = "fe_corr_pearson" 
    FE_CORR_KENDALL = "fe_corr_kendall"
    FE_CORR_SPEARMAN = "fe_corr_spearman"

    ########## Value node pipelines ##########
    CHECK_VN = "check_vn"

    # ML
    ML_LOGISTIC_REG = "ml_logistic_reg"
    ML_RANDOM_FOREST = "ml_random_forest"
    ML_SGD = "ml_sgd"

    ########## Data structure pipelines ##########
    # clustering
    DS_DENSITY_CLUSTERING = "ds_density_clustering"


def is_feature_engineering_pipeline(pipeline_name: PipelineNames):
    """
    Return True if the pipeline is a feature engineering pipeline.
    """
    return pipeline_name in [
        PipelineNames.FE_UNIVARIATE,
        PipelineNames.FE_CORR_PEARSON,
        PipelineNames.FE_CORR_KENDALL,
        PipelineNames.FE_CORR_SPEARMAN,
    ]

def is_value_node_ml_pipeline(pipeline_name: PipelineNames):
    """
    Value node pipelines expect value node dataset.
    """
    return pipeline_name in [
        PipelineNames.CHECK_VN,

        # ML
        PipelineNames.ML_LOGISTIC_REG,
        PipelineNames.ML_RANDOM_FOREST,
        PipelineNames.ML_SGD,
    ]

def is_datastructure_ml_pipeline(pipeline_name: PipelineNames):
    """
    Data structure pipelines expect data structure dataset.
    """
    return pipeline_name in [
        PipelineNames.DS_DENSITY_CLUSTERING,
    ]    


def print_pipeline_names():
    """
    Print the possible values of the PipelineNames.
    """
    print_enum_values(PipelineNames)


def convert_str_arg_to_pipeline_name(arg: str):
    """
    Convert a string argument to a PipelineNames.
    """
    return convert_str_arg_to_enum_member(arg, PipelineNames)


