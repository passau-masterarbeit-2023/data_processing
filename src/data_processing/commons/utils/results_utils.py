import contextlib
from datetime import datetime
import logging

from commons.results.base_result_manager import BaseResultsManager
from commons.results.base_result_writer import BaseResultWriter


@contextlib.contextmanager
def time_measure_result(
    message: str, 
    logger : logging.Logger = None,
    result_saver: BaseResultWriter | BaseResultsManager = None, 
    result_column: str = None, 
):
    """
    Measure the time elapsed since the begining of the context.
    """
    if logger is not None:
        logger.info("timer for " + message + " started")
    else:
        print("timer for " + message + " started")

    start = datetime.now()
    yield
    elapsed = datetime.now() - start
    # duration in seconds with 6 decimals
    duration_str = f"{elapsed.total_seconds():.9f}"

    message = "Time elapsed since the begining of {0}: {1} s".format(message, duration_str)
    
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    
    if result_saver is not None and result_column is not None:
        if type(result_saver) == BaseResultsManager:
            result_saver.set_result_forall(result_column, duration_str)
        elif type(result_saver) == BaseResultWriter:
            result_saver.set_result(result_column, duration_str)
        else:
            raise ValueError(f"Unknown result saver type: {type(result_saver)}")
    