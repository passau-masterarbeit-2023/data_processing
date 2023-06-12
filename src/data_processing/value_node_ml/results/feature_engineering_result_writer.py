from datetime import datetime
from commons.results.base_result_writer import BaseResultWriter

from commons.utils.utils import datetime2str
from value_node_ml.params.pipeline_params import PipelineNames

class FeatureEngineeringResultsWriter(BaseResultWriter):
    """
    This class is used to write the results of a classification pipeline to a CSV file.
    It keeps track of the headers of the CSV file and the results to write.
    It stores everything related to classification results.
    """
    ADDITIONAL_HEADERS: list[str] = [
        "feature_engineering_algorithm",
        "training_dataset_origin",
        "descending_best_column_names",
        "descending_best_column_values"
    ]

    def __init__(self, csv_file_path: str, pipeline_name: PipelineNames):
        super().__init__(csv_file_path, self.ADDITIONAL_HEADERS, pipeline_name)
        
