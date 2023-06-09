from enum import Enum

from commons.utils.enum_utils import print_enum_values, convert_str_arg_to_enum_member


class PipelineNames(Enum):
    """
    Pipeline names.
    """
    CHECK = "check"
    UNIVARIATE_FS = "univariate_fs"
    ML_LOGISTIC_REG = "ml_logistic_reg"
    ML_RANDOM_FOREST = "ml_random_forest"
    ML_SGD = "ml_sgd"


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


