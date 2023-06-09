from enum import Enum


class PipelineNames(Enum):
    """
    Pipeline names.
    """
    CHECK = "check"
    UNIVARIATE_FS = "univariate_fs"
    ML_LOGISTIC_REG = "ml_logistic_reg"
    ML_RANDOM_FOREST = "ml_random_forest"
    ML_SGD = "ml_sgd"


def print_pipeline_names() -> None:
    """
    Print the possible values of the PipelineNames.
    """
    print(f"Possible values of PipelineNames: {', '.join([pipeline.value for pipeline in PipelineNames])}")


def convert_str_arg_to_pipeline_name(arg: str) -> PipelineNames:
    """
    Convert a string argument to a PipelineNames.
    """
    if arg == PipelineNames.CHECK or arg.lower() == "check":
        return PipelineNames.CHECK
    elif arg == PipelineNames.UNIVARIATE_FS or arg.lower() == "univariate_fs":
        return PipelineNames.UNIVARIATE_FS
    elif arg == PipelineNames.ML_LOGISTIC_REG or arg.lower() == "ml_logistic_reg":
        return PipelineNames.ML_LOGISTIC_REG
    elif arg == PipelineNames.ML_RANDOM_FOREST or arg.lower() == "ml_random_forest":
        return PipelineNames.ML_RANDOM_FOREST
    elif arg == PipelineNames.ML_SGD or arg.lower() == "ml_sgd":
        return PipelineNames.ML_SGD
    else:
        print_pipeline_names()
        raise ValueError(f"Unknown pipeline name: {arg}.")
