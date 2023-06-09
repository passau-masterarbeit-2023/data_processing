# direct raw access to params
import sys
import argparse

from value_node_ml.params.balancing_params import BalancingStrategies
from value_node_ml.params.pipeline_params import PipelineNames

# wrapped program flags
class CLIArguments:
    args: argparse.Namespace

    def __init__(self) -> None:
        self.__log_raw_argv()
        self.__parse_argv()
    
    def __log_raw_argv(self) -> None:
        print("Passed program params:")
        for i in range(len(sys.argv)):
            print("param[{0}]: {1}".format(
                i, sys.argv[i]
            ))
    
    def __parse_argv(self) -> None:
        """
        python src/feature_engineering/main.py [ARGUMENTS ...]

        Parse program arguments.
            -w max ml workers (threads for ML threads pool, -1 for illimited)
            -d debug
            -p pipelines
            -otr origins training
            -ots origins testing
            -b balancing strategy
            -h help
            --profile launch profiler
        
        usage example:
            python3 main.py -m GRID_SEARCH_CV -b OVER -t /home/onyr/Documents/code/phdtrack/phdtrack_data/Training/Training/scp/V_7_8_P1/16 -e /home/onyr/Documents/code/phdtrack/phdtrack_data/Validation/Validation/scp/V_7_8_P1/16 -d False -v 2
        """
        parser = argparse.ArgumentParser(description='Program [ARGUMENTS]')
        parser.add_argument(
            '-d',
            '--debug', 
            action='store_true',
            help="debug, True or False"
        )
        parser.add_argument(
            '-w',
            '--max_ml_workers', 
            type=int, 
            default=None,
            help="max ml workers (threads for ML threads pool, -1 for illimited)"
        )
        parser.add_argument(
            '-p',
            '--pipelines',
            type=str,
            nargs='*',
            default=None,
            help="List of pipelines to run: " + str(list(map(lambda x: x.name.lower(), PipelineNames)))
        )
        parser.add_argument(
            '-otr',
            '--origins_training',
            type=str,
            nargs='*',
            default=None,
            help="Data origin (training, validation, testing) for training"
        )
        parser.add_argument(
            '-ots',
            '--origins_testing',
            type=str,
            nargs='*',
            default=None,
            help="Data origin (training, validation, testing) for testing"
        )
        parser.add_argument(
            '-b',
            '--balancing_strategy',
            type=str,
            default=None,
            help="Balancing strategy for training data. Possible strategies: " + str(list(map(lambda x: x.name.lower(), BalancingStrategies)))
        )
        parser.add_argument(
            '--profile',
            action='store_true',
            help="Launch profiler"
        )

        # save parsed arguments
        self.args = parser.parse_args()