import contextlib
from commons.results.base_result_manager import BaseResultsManager
from value_node_ml.results.result_writer import ClassificationResultsWriter
from value_node_ml.results.result_writer import ClassificationResultsWriter
from .data_origin import DataOriginEnum, convert_str_arg_to_data_origin, print_data_origin_enum
from ..cli import CLIArguments
from commons.utils.utils import DATETIME_FORMAT, datetime2str
from value_node_ml.params.pipeline_params import PipelineNames, convert_str_arg_to_pipeline_name, print_pipeline_names

from dataclasses import dataclass
import dotenv
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ProgramParams:
    """
    Wrapper class for program parameters.
    """
    cli_args: CLIArguments
    results_manager: BaseResultsManager[PipelineNames, ClassificationResultsWriter]
    __paths_vars_to_check: list[str] = []

    ### cli args
    pipelines: list[PipelineNames] | None = None
    data_origins_training: set[DataOriginEnum] | None = None
    data_origins_testing: set[DataOriginEnum] | None = None
    use_batch: bool = False
    
    ### env vars
    # NOTE: all None values NEED to be overwritten by the .env file
    
    # default values
    MAX_ML_WORKERS: int = None
    DEBUG: bool = None

    # data
    CSV_DATA_SAMPLES_AND_LABELS_DIR_PATH: str | None = None
    __paths_vars_to_check.append("CSV_DATA_SAMPLES_AND_LABELS_DIR_PATH")

    # results
    CSV_CLASSIFICATION_RESULTS_PATH: str = None
    __paths_vars_to_check.append("CSV_CLASSIFICATION_RESULTS_PATH")

    # logger
    COMMON_LOGGER_DIR_PATH: str = None
    __paths_vars_to_check.append("COMMON_LOGGER_DIR_PATH")

    RESULTS_LOGGER_DIR_PATH: str = None
    __paths_vars_to_check.append("RESULTS_LOGGER_DIR_PATH")

    COMMON_LOGGER = logging.getLogger("common_logger")
    RESULTS_LOGGER = logging.getLogger("results_logger")


    SAVE_RESULT_LOGS: bool = True

    def __init__(
            self, 
            load_program_argv : bool = True, 
            debug : bool = False,
            **kwargs
    ):
        """
        Program parameters. 
        If load_program_argv is True, the program parameters are loaded 
        from the command line arguments.
        Otherwise, the program parameters are loaded from the default values.
        if generate_important_log_file is True, the program will generate 
        an important log file with the current date and time.
        otherwise, the program will use the default log file.

        debug and generate_important_log_file parameter is used 
        only if load_program_argv is True.
        """
        if load_program_argv:
            self.__parse_program_argv()
        else:
            self.DEBUG = debug
            self.SAVE_RESULT_LOGS = False

        self.__load_env()
        self.__check_all_paths()

        self.__construct_log()
        self.__log_program_params()

        # keep results
        self.results_manager = BaseResultsManager[PipelineNames, ClassificationResultsWriter](
            self.CSV_CLASSIFICATION_RESULTS_PATH, ClassificationResultsWriter
        )
    
    @classmethod
    def __load_env(self):
        """
        Load environment variables from .env file.
        Overwrite default values with values from .env file if they are defined there.
        """
        # determine project .env file, using the current python file location
        # and check recursively in parent directories for the first encountered .env file
        tmp_folder = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(tmp_folder + "/.env"):
            tmp_folder = os.path.dirname(tmp_folder)
            if tmp_folder == "/":
                print("ERROR: .env file not found")
                exit(1)
        self.PROJECT_BASE_DIR = tmp_folder + "/"


        # Load environment variables from .env file
        dotenv.load_dotenv(dotenv_path=self.PROJECT_BASE_DIR + ".env")

        # Overwrite default values with values from .env file if they are defined there
        # NOTE: cls.__annotations__ is a dictionary where the keys are the names of 
        #   the class variables and the values are their types
        for variable in self.__annotations__.keys():
            env_value = os.getenv(variable)
            if env_value is not None:
                # Convert to the appropriate type
                if self.__annotations__[variable] == bool:
                    env_value = env_value.lower() == 'true'
                elif self.__annotations__[variable] == int:
                    env_value = int(env_value)
                elif self.__annotations__[variable] == list:
                    env_value = env_value.split(',')
                setattr(self, variable, env_value)


    def __is_running_under_pytest(self):
        """
        Check whether the code is running under pytest.
        """
        return 'pytest' in sys.modules

    def __check_all_paths(self):
        """
        Check if all paths are valid.
        """
        for path_var_name in self.__paths_vars_to_check:
            path = getattr(self, path_var_name)
            if not os.path.exists(path):
                print("Program paths are NOT OK. Error in var: %s" % path_var_name)
                print("%s: %s" % (path_var_name, path))
                exit(1)
        
        print("✅ Program paths are OK.")
            
    
    def __parse_program_argv(self):
        """
        Parse program arguments.
        WARN: Do NOT parse program argv if running under pytest.
        """
        if not self.__is_running_under_pytest():
            self.__load_program_argv()
            self.__consume_program_argv()
    
    def __load_program_argv(self):
        """
        Load given program arguments.
        """
        self.cli_args: CLIArguments = CLIArguments()
    
    def __consume_program_argv(self):
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

    def __construct_log(self):
        """
        construct logger. Must be call after __load_program_argv()
        """
        # common formater
        common_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt=DATETIME_FORMAT.replace("_%f", "")
        )

        # ----------------- common logger -----------------
        # create common logger
        self.COMMON_LOGGER.setLevel(logging.DEBUG)

        # add RotatingFileHandler to common logger
        common_log_file_path = self.COMMON_LOGGER_DIR_PATH + "/common_log.log"
        common_log_file_handler = RotatingFileHandler(
            common_log_file_path, maxBytes=50000000, backupCount=5
        )
        logging_level = logging.DEBUG if self.DEBUG else logging.INFO
        common_log_file_handler.setLevel(logging_level)
        common_log_file_handler.setFormatter(common_formatter)
        self.COMMON_LOGGER.addHandler(common_log_file_handler)

        # add console handler to common logger
        common_log_console_handler = logging.StreamHandler(stream=sys.stdout)
        common_log_console_handler.setLevel(logging.INFO)
        common_log_console_handler.setFormatter(common_formatter)
        self.COMMON_LOGGER.addHandler(common_log_console_handler)

        # ----------------- results logger -----------------
        # create results logger
        self.RESULTS_LOGGER.setLevel(logging.DEBUG)

        # Result logger using file handler
        if self.SAVE_RESULT_LOGS:
            results_log_file_path = self.RESULTS_LOGGER_DIR_PATH + "/" + datetime2str(datetime.now()) + "_results.log"
        else:
            results_log_file_path = common_log_file_path
        results_log_file_handler = logging.FileHandler(results_log_file_path)
        results_log_file_handler.setLevel(logging.DEBUG)
        results_log_file_handler.setFormatter(common_formatter)
        self.RESULTS_LOGGER.addHandler(results_log_file_handler)

        # Result logger using console handler
        results_log_console_handler = logging.StreamHandler(stream=sys.stdout)
        results_log_console_handler.setLevel(logging.DEBUG)
        results_log_console_handler.setFormatter(common_formatter)
        self.RESULTS_LOGGER.addHandler(results_log_console_handler)
    
    def __log_program_params(self):
        """
        Log given program arguments.
        """
        self.RESULTS_LOGGER.info("Program params:   [see below]")
        # complete list of program params below
        self.RESULTS_LOGGER.info("\tdebug: %s" % self.DEBUG)
        self.RESULTS_LOGGER.info("\tmax_ml_workers: %s" % self.MAX_ML_WORKERS)