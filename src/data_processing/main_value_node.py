from datetime import datetime
from pathlib import Path
from commons.utils.utils import DATETIME_FORMAT, time_measure_result
from value_node_ml.data_loading.data_types import SamplesAndLabels
from value_node_ml.data_loading.data_loading import load
from commons.params.data_origin import DataOriginEnum
from value_node_ml.params.pipeline_params import print_pipeline_names
from value_node_ml.pipelines.pipelines import PIPELINE_NAME_TO_FUNCTION
from value_node_ml.params.params import ProgramParams


import cProfile
import pandas as pd

# run: python src/feature_engineering/main.py
def main(params: ProgramParams):

    # consume program argv pipelines and run them
    if params.pipelines is None:
        # no pipelines to run
        params.COMMON_LOGGER.warning(f"Pipelines is None (params.PIPELINES: {params.pipelines})")
        print_pipeline_names()
        exit(1)

    if len(params.pipelines) == 0:
        params.COMMON_LOGGER.warning(f"No pipelines to run (params.PIPELINES: {params.pipelines})")
        print_pipeline_names()
        exit(1)

    if params.data_origins_training is None or len(params.data_origins_training) == 0:
        params.COMMON_LOGGER.warning(f"No training data origins (params.DATA_ORIGINS_TRAINING: {params.data_origins_training})")
        exit(1)
    
    # check that params.DATA_ORIGINS_TRAINING and params.DATA_ORIGINS_TESTING are disjoint
    if params.data_origins_testing is not None and len(params.data_origins_testing) > 0:
        if len(params.data_origins_training.intersection(params.data_origins_testing)) > 0:
            params.COMMON_LOGGER.warning(f"Training and testing data origins are not disjoint (params.DATA_ORIGINS_TRAINING: {params.data_origins_training}, params.DATA_ORIGINS_TESTING: {params.data_origins_testing})")
            exit(1)

    # load & clean data
    origin_to_samples_and_labels: dict[DataOriginEnum, SamplesAndLabels] = {}

    all_origins = params.data_origins_training
    if params.data_origins_testing is not None:
        all_origins = all_origins.union(params.data_origins_testing)

    with time_measure_result(
            f'load_samples_and_labels_from_all_csv_files', 
            params.RESULTS_LOGGER, 
            params.results_manager, 
            "data_loading_duration"
        ):
        for data_origin in all_origins:
            print(f"Loading data from {data_origin}")
            origin_to_samples_and_labels[data_origin] = load(
                params, 
                params.CSV_DATA_SAMPLES_AND_LABELS_DIR_PATH,
                {data_origin},
            )
    
    for pipeline_name in params.pipelines:
        # information about the current pipeline
        params.COMMON_LOGGER.info(f"Running pipeline: {pipeline_name}")
        params.results_manager.set_result_for(
            pipeline_name,
            "pipeline_name",
            pipeline_name.value,
        )

        # get current time using datetime
        start_time = datetime.now()
        params.results_manager.set_result_for(
            pipeline_name, "start_time", 
            start_time.strftime(DATETIME_FORMAT)
        )

        # run pipeline
        pipeline_function: function[ProgramParams, pd.DataFrame, pd.Series] = PIPELINE_NAME_TO_FUNCTION[pipeline_name]
        with time_measure_result(f'pipeline ({pipeline_name})', params.RESULTS_LOGGER):
            pipeline_function(params, origin_to_samples_and_labels)
        
        # get current time using time 
        end_time = datetime.now()
        params.results_manager.set_result_for(
            pipeline_name, 
            "end_time", 
            end_time.strftime(DATETIME_FORMAT)
        )

        # compute duration
        duration = end_time - start_time
        duration_str = f"{duration.total_seconds():.9f}"
        params.results_manager.set_result_for(
            pipeline_name,
            "duration",
            duration_str,
        )
    
        # write results
        params.results_manager.write_results_for(pipeline_name)
                
        
def profiling_main(params: ProgramParams):
    import cProfile
    import pstats
    import datetime

    FUNCTION_TO_PROFILE = main
    timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT)

    # profile code
    profiler = cProfile.Profile()
    profiler.runctx(
        FUNCTION_TO_PROFILE.__name__ + "(params)", 
        globals(), locals(), 
    )

    # create and sort stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    # write stats to a file
    stat_file_path = Path(params.PROFILING_LOGS_DIR_PATH).joinpath(
        f'sorted_profiling_output_{timestamp}.txt'
    )

    with open(stat_file_path, 'w') as f:
        stats.stream = f
        stats.print_stats()




if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    # run main
    if params.PROFILE:
        profiling_main(params)
    else:
        main(params)

    #cProfile.run('main()')
