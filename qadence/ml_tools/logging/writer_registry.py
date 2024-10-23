from torch.utils.tensorboard import SummaryWriter
from qadence.types import ExperimentTrackingTool
import os
from uuid import uuid4
import importlib

class BaseWriter:
    def open(self, config, iteration=None):
        raise NotImplementedError("Writers must implement an open method.")
    
    def close(self):
        raise NotImplementedError("Writers must implement a close method.")

class TensorBoardWriter(BaseWriter):
    def __init__(self):
        self.writer = None

    def open(self, config, iteration=None):
        # Optional purge_step handling
        if purge_step is not None:
            self.writer = SummaryWriter(log_dir=str(config.folder), purge_step=iteration)
        else:
            self.writer = SummaryWriter(log_dir=str(config.folder))  
        return self.writer

    def close(self):
        if self.writer:
            self.writer.close()

class MLFlowWriter(BaseWriter):
    def __init__(self):
        self.run = None

    def open(self, config, iteration=None):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", str(uuid4()))
        run_name = os.getenv("MLFLOW_RUN_NAME", str(uuid4()))

        # Set up MLFlow tracking
        mlflow = importlib.import_module("mlflow")
        mlflow.set_tracking_uri(tracking_uri)
        exp_filter_string = f"name = '{experiment_name}'"
        if not mlflow.search_experiments(filter_string=exp_filter_string):
            mlflow.create_experiment(name=experiment_name)

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name, nested=False)
        
        return mlflow  

    def close(self):
        if self.run:
            mlflow = importlib.import_module("mlflow")
            mlflow.end_run()


# Writer Registry
WRITER_REGISTRY = {
    ExperimentTrackingTool.TENSORBOARD: TensorBoardWriter(),
    ExperimentTrackingTool.MLFLOW: MLFlowWriter()
}

def get_writer(tracking_tool):
    return WRITER_REGISTRY.get(tracking_tool)
