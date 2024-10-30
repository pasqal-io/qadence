from __future__ import annotations

import os
from uuid import uuid4
from logging import getLogger
from typing import Any, Callable, Union

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matplotlib.figure import Figure
from qadence.ml_tools.data import DictDataLoader
from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)

# Type aliases
PlottingFunction = Callable[[Module, int], tuple[str, Figure]]
InputData = Union[Tensor, dict[str, Tensor]]


class BaseWriter:
    """Abstract base class for writers."""

    def open(self, config, iteration: int = None):
        raise NotImplementedError("Writers must implement an open method.")

    def close(self):
        raise NotImplementedError("Writers must implement a close method.")

    def write(self, loss: float | None, metrics: dict, iteration: int) -> None:
        raise NotImplementedError("Writers must implement a write method.")

    def log_hyperparams(self, hyperparams: dict, metrics: dict) -> None:
        raise NotImplementedError("Writers must implement a log_hyperparams method.")

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        raise NotImplementedError("Writers must implement a plot method.")

    def log_model(
        self,
        model: Module,
        dataloader: Union[None, DataLoader, DictDataLoader],
    ) -> None:
        raise NotImplementedError("Writers must implement a log_model method.")
    
    def print_metrics(self, loss: float | None, metrics: dict, iteration: int) -> None:
        msg = " ".join(
            [f"Iteration {iteration: >7} | Loss: {loss:.7f} -"]
            + [f"{k}: {v.item():.7f}" for k, v in metrics.items()]
        )
        print(msg)


class TensorBoardWriter(BaseWriter):
    """Writer class for TensorBoard."""

    def __init__(self):
        self.writer = None

    def open(self, config, iteration: int = None):
        log_dir = str(config.log_folder)
        if iteration is not None:
            self.writer = SummaryWriter(log_dir=log_dir, purge_step=iteration)
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer

    def close(self):
        if self.writer:
            self.writer.close()

    def write(self, loss: float = None, metrics: dict = {}, iteration: int = 0) -> None:
        if loss is not None:
            self.writer.add_scalar("loss", loss, iteration)
        for key, arg in metrics.items():
            self.writer.add_scalar(key, arg, iteration)

    def log_hyperparams(self, hyperparams: dict, metrics: dict) -> None:
        self.writer.add_hparams(hyperparams, metrics)

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        for pf in plotting_functions:
            descr, fig = pf(model, iteration)
            self.writer.add_figure(descr, fig, global_step=iteration)

    def log_model(
        self,
        model: Module,
        dataloader: Union[None, DataLoader, DictDataLoader],
    ) -> None:
        logger.warning(
            "Model logging is not supported by TensorBoard in this implementation."
        )


class MLFlowWriter(BaseWriter):
    """Writer class for MLflow."""

    def __init__(self):
        self.run = None
        self.mlflow = None

    def open(self, config, iteration: int = None):
        import mlflow

        self.mlflow = mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", str(uuid4()))
        run_name = os.getenv("MLFLOW_RUN_NAME", str(uuid4()))

        self.mlflow.set_tracking_uri(tracking_uri)

        # Create or get the experiment
        exp_filter_string = f"name = '{experiment_name}'"
        experiments = self.mlflow.search_experiments(filter_string=exp_filter_string)
        if not experiments:
            self.mlflow.create_experiment(name=experiment_name)

        self.mlflow.set_experiment(experiment_name)
        self.run = self.mlflow.start_run(run_name=run_name, nested=False)

        return self.mlflow

    def close(self):
        if self.run:
            self.mlflow.end_run()

    def write(self, loss: float | None, metrics: dict, iteration: int) -> None:
        if loss is not None:
            self.mlflow.log_metrics({"loss": float(loss)}, step=iteration)
        self.mlflow.log_metrics(metrics, step=iteration)

    def log_hyperparams(self, hyperparams: dict, metrics: dict) -> None:
        self.mlflow.log_params(hyperparams)
        # Optionally log initial metrics
        self.mlflow.log_metrics(metrics)

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        for pf in plotting_functions:
            descr, fig = pf(model, iteration)
            # Save figure to a temporary file
            fig_path = f"{descr}_{iteration}.png"
            fig.savefig(fig_path)
            # Log the figure as an artifact
            self.mlflow.log_artifact(fig_path)
            # Remove the temporary file
            os.remove(fig_path)

    def log_model(
        self,
        model: Module,
        dataloader: DataLoader | DictDataLoader | None,
    ) -> None:
        signature = None
        if dataloader is not None:
            xs: InputData
            xs, *_ = next(iter(dataloader))
            preds = model(xs)
            if isinstance(xs, Tensor):
                xs = xs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
            elif isinstance(xs, dict):
                xs = {key: val.detach().cpu().numpy() for key, val in xs.items()}
                preds = {key: val.detach().cpu().numpy() for key, val in preds.items()}
            try:
                from mlflow.models import infer_signature

                signature = infer_signature(xs, preds)
            except ImportError:
                logger.warning(
                    "MLflow's infer_signature is not available. Please install mlflow."
                )

        self.mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)

# Writer registry
WRITER_REGISTRY = {
    ExperimentTrackingTool.TENSORBOARD: TensorBoardWriter,
    ExperimentTrackingTool.MLFLOW: MLFlowWriter,
}


def get_writer(tracking_tool: ExperimentTrackingTool) -> BaseWriter:
    writer_class = WRITER_REGISTRY.get(tracking_tool)
    if writer_class:
        return writer_class()
    else:
        raise ValueError(f"Unsupported tracking tool: {tracking_tool}")
