from datetime import datetime
import json

# utils constants
DATETIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"


def str2bool(string: str) -> bool:
    return json.loads(string.lower())


def str2enum(string: str, enum_type: type):
    return enum_type[string.upper()]


def datetime2str(datetime: datetime):
    """
    Return a string representation of the given datetime.
    NB: The %f is the microseconds.
    """
    return datetime.strftime(DATETIME_FORMAT)



