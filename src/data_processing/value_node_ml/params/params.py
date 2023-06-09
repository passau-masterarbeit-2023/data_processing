from commons.params.base_program_params import BaseProgramParams
from commons.params.data_origin import convert_str_arg_to_data_origin
from commons.results.base_result_manager import BaseResultsManager
from value_node_ml.results.result_writer import ClassificationResultsWriter
from value_node_ml.results.result_writer import ClassificationResultsWriter
from ..cli import CLIArguments
from value_node_ml.params.pipeline_params import PipelineNames, convert_str_arg_to_pipeline_name


class ProgramParams(BaseProgramParams):
    """
    Wrapper class for program parameters.
    """
    results_manager: BaseResultsManager[PipelineNames, ClassificationResultsWriter]
    pipelines: list[PipelineNames]
    cli_args: CLIArguments
    
    ### env vars
    # NOTE: all None values NEED to be overwritten by the .env file

    # data
    CSV_DATA_SAMPLES_AND_LABELS_DIR_PATH: str 

    # results
    CSV_CLASSIFICATION_RESULTS_PATH: str

    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        super().__init__(load_program_argv, debug)

        # keep results
        self.results_manager = BaseResultsManager[PipelineNames, ClassificationResultsWriter](
            self.CSV_CLASSIFICATION_RESULTS_PATH, ClassificationResultsWriter
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
            # NOTE: please, let DATA_ORIGINS_TESTING to none, such that we can split the data in the pipeline if needed.
        
        if self.cli_args.args.pipelines is not None:
            try:
                self.pipelines = set(map(convert_str_arg_to_pipeline_name, self.cli_args.args.pipelines))
                assert isinstance(self.pipelines, set)
            except ValueError:
                    print(f"ERROR: Invalid pipeline name: {self.cli_args.args.pipelines}")
                    exit(1)
        # No if here, batch is either True or False
        self.use_batch = self.cli_args.args.batch
        assert isinstance(self.use_batch, bool)
