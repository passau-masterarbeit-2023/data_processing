from enum import Enum
from typing import Optional
from value_node_ml.params.pipeline_params import PipelineNames
from .base_result_writer import BaseResultWriter

from typing import TypeVar, Generic, Type

ResultWriter = TypeVar('ResultWriter', bound=BaseResultWriter)  # CustomResultWriter should be a subtype of BaseResultWriter
PipelineNamesEnum = TypeVar('PipelineNamesEnum', bound=Enum)  # PipelineNamesEnum should be a subtype of Enum

class BaseResultsManager(Generic[PipelineNamesEnum, ResultWriter]):
    """
    Manager of different result writers (one per pipeline).
    """
    classification_result_writer_dict: dict[PipelineNamesEnum, ResultWriter]
    csv_classification_results_path: str

    def __init__(self, csv_classification_results_path: str, ResultWriterType: Type[BaseResultWriter]):
        self.csv_classification_results_path = csv_classification_results_path
        
        # result keepers
        self.__create_results_keepers(ResultWriterType)

    def __create_results_keepers(self, ResultWriterType: Type[BaseResultWriter]):
        """
        Create results keepers.
        """
        self.classification_result_writer_dict = {}
        for pipeline_name in PipelineNames:
            self.classification_result_writer_dict[pipeline_name] = ResultWriterType(
                self.csv_classification_results_path,
                pipeline_name ,
            )
    
    def set_result_forall(
        self, field: str, value: Optional[str]    
    ) -> None:
        """
        Set a result for all result keepers.
        """
        for classification_result_writer in self.classification_result_writer_dict.values():
            classification_result_writer.set_result(field, value)
    
    def set_result_for(
        self, pipeline_name: PipelineNamesEnum, field: str, value: Optional[str]    
    ) -> None:
        """
        Set a result for a specific result keeper.
        """
        self.classification_result_writer_dict[pipeline_name].set_result(field, value)

    def write_results_forall(self) -> None:
        """
        Write results for all result keepers.
        """
        for classification_result_writer in self.classification_result_writer_dict.values():
            classification_result_writer.write_results()

    def write_results_for(self, pipeline_name: PipelineNamesEnum) -> None:
        """
        Write results for a specific result keeper.
        """
        self.classification_result_writer_dict[pipeline_name].write_results()
    
    def get_result_writer_for(self, pipeline_name: PipelineNamesEnum) -> ResultWriter:
        """
        Get a specific result keeper.
        """
        return self.classification_result_writer_dict[pipeline_name]
