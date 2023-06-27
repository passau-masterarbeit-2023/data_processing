from commons.params.base_program_params import BaseProgramParams
from commons.params.data_origin import convert_str_arg_to_data_origin
from commons.results.base_result_manager import BaseResultsManager
from commons.params.app_params import AppName
from value_node_ml.params.dataset_loading_params import DatasetLoadingPossibilities, check_data_structure_dataset_params, check_value_node_dataset_params, convert_str_arg_to_dataset
from value_node_ml.results.feature_engineering_result_writer import FeatureEngineeringResultsWriter
from value_node_ml.params.balancing_params import BalancingStrategies, convert_str_arg_to_balancing_strategy
from value_node_ml.results.classification_result_writer import ClassificationResultsWriter
from value_node_ml.params.pipeline_params import PipelineNames, convert_str_arg_to_pipeline_name, is_datastructure_ml_pipeline, is_feature_engineering_pipeline, is_value_node_ml_pipeline
from value_node_ml.results.classification_result_writer import ClassificationResultsWriter
from ..cli import CLIArguments


class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    ml_results_manager: BaseResultsManager[PipelineNames, ClassificationResultsWriter]
    fe_results_manager: BaseResultsManager[PipelineNames, FeatureEngineeringResultsWriter]

    pipelines: set[PipelineNames]
    cli_args: CLIArguments
    
    ### env vars
    # NOTE: all CAPITAL_PARAM_VALUES values NEED to be overwritten by the .env file
    # NOTE: lowercase values are from the CLI

    # data
    columns_to_keep: set[str]
    CSV_DATASET_SAMPLES_AND_LABELS_DIR_PATH: str
    CSV_DATASET_DATA_STRUCTURE_DIR_PATH: str

    # results
    CSV_CLASSIFICATION_RESULTS_PATH: str
    CSV_FEATURE_ENGINEERING_RESULTS_PATH: str
    FEATURE_CORRELATION_MATRICES_RESULTS_DIR_PATH: str

    # ML
    balancing_strategy: BalancingStrategies

    # feature engineering
    FEATURE_ENGINEERING_NB_KEEP_BEST_COLUMNS: int

    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        super().__init__(AppName.FEATURE_EMBEDDING_ML, load_program_argv, debug)

        # cheking stage
        self.__check_params_relevance()

        # keep results
        self.__results_manager_init()

        # to be done last
        self._log_program_params()
    
    def __results_manager_init(self):
        """
        Initialize results manager, and start keeping results-related information.
        """
        # create ML results manager
        self.ml_results_manager = BaseResultsManager[PipelineNames, ClassificationResultsWriter](
            self.CSV_CLASSIFICATION_RESULTS_PATH, ClassificationResultsWriter
        )

        # create FE results manager
        print("CSV_FEATURE_ENGINEERING_RESULTS_PATH", self.CSV_FEATURE_ENGINEERING_RESULTS_PATH)
        self.fe_results_manager = BaseResultsManager[PipelineNames, FeatureEngineeringResultsWriter](
            self.CSV_FEATURE_ENGINEERING_RESULTS_PATH, FeatureEngineeringResultsWriter
        )

        # save data origins on ML results manager
        self.ml_results_manager.set_result_forall(
            "training_dataset_origin",
            " ".join([origin.value for origin in self.data_origins_training])
        )
        if self.data_origins_testing is not None:
            self.ml_results_manager.set_result_forall(
                "testing_dataset_origin", 
                " ".join([origin.value for origin in self.data_origins_testing])
            )
        
        self.set_result_forall(
            "random_seed",
            str(self.RANDOM_SEED)
        )

        # save info for FE results manager
        self.fe_results_manager.set_result_forall(
            "training_dataset_origin",
            " ".join([origin.value for origin in self.data_origins_training])
        )
    
    
    def _load_program_argv(self):
        """
        Load given program arguments.
        """
        self.cli_args: CLIArguments = CLIArguments()
    
    def _consume_program_argv(self):
        """
        Consume given program arguments.
        """
        if self.cli_args.args.debug is not None:
            self.DEBUG = self.cli_args.args.debug
            assert isinstance(self.DEBUG, bool)

        if self.cli_args.args.max_ml_workers is not None:
            self.MAX_ML_WORKERS = int(self.cli_args.args.max_ml_workers)
            assert isinstance(self.MAX_ML_WORKERS, int)

        if self.cli_args.args.origins_training is not None:
            try:
                self.data_origins_training = set(map(convert_str_arg_to_data_origin, self.cli_args.args.origins_training))
                assert isinstance(self.data_origins_training, set)
            except ValueError:
                print(f"ERROR: Invalid data origin training: {self.cli_args.args.origins_training}")
                exit(1)
        
        if self.cli_args.args.origins_testing is not None:
            try:
                self.data_origins_testing = set(map(convert_str_arg_to_data_origin, self.cli_args.args.origins_testing))
                assert isinstance(self.data_origins_testing, set)
            except ValueError:
                print(f"ERROR: Invalid data origin testing: {self.cli_args.args.origins_testing}")
                exit(1)
            # NOTE: when DATA_ORIGINS_TESTING to none, we can split the data in the pipeline if needed.
        
        if self.cli_args.args.pipelines is not None:
            try:
                self.pipelines = set(map(convert_str_arg_to_pipeline_name, self.cli_args.args.pipelines))
                assert isinstance(self.pipelines, set)

            except ValueError:
                    print(f"ERROR: Invalid pipeline name: {self.cli_args.args.pipelines}")
                    exit(1)

        try:
            if self.cli_args.args.balancing_strategy is not None:
                self.balancing_strategy = convert_str_arg_to_balancing_strategy(self.cli_args.args.balancing_strategy)
            else:
                self.balancing_strategy = BalancingStrategies.NO_BALANCING
            
            assert isinstance(self.balancing_strategy, BalancingStrategies)
        except ValueError:
            print(f"ERROR: Invalid balancing strategy: {self.cli_args.args.balancing_strategy}")
            exit(1)
        
        if self.cli_args.args.profiling is not None:
            self.PROFILING = self.cli_args.args.profiling
            assert isinstance(self.PROFILING, bool)
        
        if self.cli_args.args.columns_to_keep is not None:
            try:
                self.columns_to_keep = set(self.cli_args.args.columns_to_keep)
                assert isinstance(self.columns_to_keep, set)
            except ValueError:
                print(f"ERROR: Invalid columns to keep: {self.cli_args.args.columns_to_keep}")
                exit(1)
        else:
            self.columns_to_keep = None
        
        if self.cli_args.args.dataset is not None:
            try:
                self.dataset = convert_str_arg_to_dataset(self.cli_args.args.dataset)
                assert isinstance(self.dataset, DatasetLoadingPossibilities)
            except ValueError:
                print(f"ERROR: Invalid dataset: {self.cli_args.args.dataset}")
                exit(1)
    
    
    def __check_params_relevance(self):
        """
        Some parameters are only relevant when used with other parameters.
        Some parameter values should not be used when other parameters are used.
        The following function check the relevance of the parameters.
        """
        for pipeline in self.pipelines:
            # NOTE: feature engineering pipelines are common to both data structures and value nodes
            if not is_feature_engineering_pipeline(pipeline):
                if is_value_node_ml_pipeline(pipeline):
                    check_value_node_dataset_params(self.dataset)
                elif is_datastructure_ml_pipeline(pipeline):
                    check_data_structure_dataset_params(self.dataset)

    # result wrappers
    def save_results_to_csv(self, pipeline_name: PipelineNames):
        """
        Save results to CSV files.
        """
        if is_feature_engineering_pipeline(pipeline_name):
            self.fe_results_manager.save_results_for(pipeline_name)
        else:
            self.ml_results_manager.save_results_for(pipeline_name)
    
    def set_result_for(self, pipeline_name: PipelineNames, column_name: str, value: str):
        """
        Set a result for a given pipeline.
        """
        if is_feature_engineering_pipeline(pipeline_name):
            self.fe_results_manager.set_result_for(pipeline_name, column_name, value)
        else:
            self.ml_results_manager.set_result_for(pipeline_name, column_name, value)

    def set_result_forall(self, column_name: str, value: str):
        """
        Set a result for all pipelines.
        """
        self.ml_results_manager.set_result_forall(column_name, value)
        self.fe_results_manager.set_result_forall(column_name, value)