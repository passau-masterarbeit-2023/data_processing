from abc import ABC, abstractmethod # for abstract classes
from .data_origin import DataOriginEnum
from commons.utils.utils import DATETIME_FORMAT, datetime2str

import dotenv
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


class BaseProgramParams(ABC):
    """
    Program parameters.
    This is a base class that contains the common parameters for all programs,
    like loggers, automatic path checking, etc.
    """

    ### cli args
    data_origins_training: set[DataOriginEnum]
    data_origins_testing: set[DataOriginEnum]
    
    ### env vars
    # NOTE: all None values NEED to be overwritten by the .env file
    
    # default values
    MAX_ML_WORKERS: int
    DEBUG: bool

    # logger
    COMMON_LOGGER_DIR_PATH: str

    RESULTS_LOGGER_DIR_PATH: str

    COMMON_LOGGER: logging.Logger 
    RESULTS_LOGGER: logging.Logger
    SAVE_RESULT_LOGS: bool

    # profiling
    PROFILE: bool
    PROFILING_LOGS_DIR_PATH: str

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
        self.__init_default_values()

        if load_program_argv:
            self.__parse_program_argv()
        else:
            self.DEBUG = debug
            self.SAVE_RESULT_LOGS = False

        self.__load_env()
        self.__check_all_paths()

        self.__construct_log()
        self.__log_program_params()

    
    def __init_default_values(self):
        """
        Init values as instance variables.
        """
        self.DEBUG = False
        self.MAX_ML_WORKERS = 10
        self.COMMON_LOGGER = logging.getLogger("common_logger")
        self.RESULTS_LOGGER = logging.getLogger("results_logger")

    
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

        # PYTHON MAGIC to get all the class variables
        all_annotations = {}

        # Iterate through the classes in the MRO in reverse order
        for cls in reversed(self.__class__.mro()):
            # Skip 'object', which has no annotations
            if cls is object:
                continue
            
            # Update the dictionary with the annotations from this class
            all_annotations.update(cls.__dict__.get('__annotations__', {}))

        # Now, 'all_annotations' should contain annotations from both the base and child classes.

        # Overwrite default values with values from .env file if they are defined there
        # NOTE: cls.__annotations__ is a dictionary where the keys are the names of 
        #   the class variables and the values are their types
        for variable in all_annotations.keys():
            env_value = os.getenv(variable)
            if env_value is not None:
                # Convert to the appropriate type
                if all_annotations[variable] == bool:
                    env_value = env_value.lower() == 'true'
                elif all_annotations[variable] == int:
                    env_value = int(env_value)
                elif all_annotations[variable] == list:
                    env_value = env_value.split(',')
                elif all_annotations[variable] == str:
                    env_value = env_value
                setattr(self, variable, env_value)
                print("---> env var: %s, val: %s" % (variable, env_value))
        
        print("✅ Environment variables loaded.")


    def __is_running_under_pytest(self):
        """
        Check whether the code is running under pytest.
        """
        return 'pytest' in sys.modules

    def __check_all_paths(self):
        """
        Check if all paths are valid.
        """
        # get all parameters that are potentially paths
        __paths_vars_to_check = []
        for name, value in vars(self).items():
            if "PATH" in name:
                __paths_vars_to_check.append(name)

        # check if all paths exist
        for path_var_name in __paths_vars_to_check:
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
            self._load_program_argv()
            self._consume_program_argv()
    
    @abstractmethod # to implement in child
    def _load_program_argv(self):
        """
        Load given program arguments.
        """
        pass
    
    @abstractmethod # to implement in child
    def _consume_program_argv(self):
        """
        Consume given program arguments.
        """
        pass

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


