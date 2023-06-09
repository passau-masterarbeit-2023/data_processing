from dataclasses import dataclass
from typing import Generator, Tuple, Union
from pandas import DataFrame, Series


SamplesAndLabels = Tuple[DataFrame, Series]

# Define the type alias
SamplesAndLabelsGenerator = Generator[SamplesAndLabels, None, None]

SamplesAndLabelsUnion = Union[SamplesAndLabels, SamplesAndLabelsGenerator]

def is_datatuple(obj) -> bool:
    return isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], DataFrame) and isinstance(obj[1], Series)

def is_datagenerator(obj) -> bool:
    return isinstance(obj, Generator)

