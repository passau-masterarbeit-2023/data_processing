import pandas as pd

from commons.utils.utils import time_measure_result
from value_node_ml.params.pipeline_params import PipelineNames
from value_node_ml.params.balancing_params import BalancingStrategies
from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.params.params import ProgramParams

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION = {
    BalancingStrategies.UNDERSAMPLING: RandomUnderSampler,
    BalancingStrategies.OVERSAMPLING: RandomOverSampler,
    BalancingStrategies.SMOTE: SMOTE,
    BalancingStrategies.ADASYN: ADASYN,
}

def resample_data(
    sampler_class, 
    params: ProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series,
) -> SamplesAndLabels:
    with time_measure_result(
            f'resample_data ({params.balancing_strategy})', 
            params.RESULTS_LOGGER, 
            params.results_manager, 
            "data_balancing_duration"
        ):
        sampler = sampler_class(random_state=params.RANDOM_SEED)
        X_res, y_res = sampler.fit_resample(samples, labels)
    return X_res, y_res

def apply_balancing(
    params: ProgramParams,
    samples: pd.DataFrame,
    labels: pd.Series,
    pipeline_name: PipelineNames,
) -> SamplesAndLabels:
    """
    Get the rebalanced data.
    """    
    if params.balancing_strategy == BalancingStrategies.NO_BALANCING:
        return samples, labels
    elif params.balancing_strategy in SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION.keys():
        params.results_manager.set_result_for(
            pipeline_name,
            "balancing_type",
            params.balancing_strategy.value
        )
        params.results_manager.set_result_for(
            pipeline_name,
            "nb_training_samples_before_balancing",
            str(len(samples))
        )
        params.results_manager.set_result_for(
            pipeline_name,
            "nb_positive_training_samples_before_balancing",
            str(len(labels[labels == 1]))
        )

        X_res, y_res = resample_data(
            SAMPLING_STRATEGY_TO_RESAMPLING_FUNCTION[params.balancing_strategy],
            params, 
            samples, 
            labels
        )

        params.results_manager.set_result_for(
            pipeline_name,
            "nb_training_samples_after_balancing",
            str(len(X_res))
        )
        params.results_manager.set_result_for(
            pipeline_name,
            "nb_positive_training_samples_after_balancing",
            str(len(y_res[y_res == 1]))
        )

        return X_res, y_res
    else:
        raise ValueError(f"Invalid balancing strategy: {params.balancing_strategy}")
