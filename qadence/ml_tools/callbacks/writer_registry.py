from __future__ import annotations

import os
from logging import getLogger
from types import ModuleType
from typing import Any, Callable, Union
from uuid import uuid4

from matplotlib.figure import Figure
from mlflow.entities import Run
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, OptimizeResult
from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)

# Type aliases
PlottingFunction = Callable[[Module, int], tuple[str, Figure]]
InputData = Union[Tensor, dict[str, Tensor]]


class BaseWriter:
    """
    Abstract base class for experiment tracking writers.

    Methods:
        open(config, iteration=None): Opens the writer and sets up the logging
            environment.
        close(): Closes the writer and finalizes any ongoing logging processes.
        print_metrics(result): Prints metrics and loss in a formatted manner.
        write(result): Writes the optimization results to the tracking tool.
        log_hyperparams(hyperparams): Logs the hyperparameters to the tracking tool.
        plot(model, iteration, plotting_functions): Logs model plots using provided
            plotting functions.
        log_model(model, dataloader): Logs the model and any relevant information.
    """

    run: Run  # [attr-defined]

    def open(self, config: TrainConfig, iteration: int = None) -> Any:
        """
        Opens the writer and prepares it for logging.

        Args:
            config: Configuration object containing settings for logging.
            iteration (int, optional): The iteration step to start logging from.
                Defaults to None.
        """
        raise NotImplementedError("Writers must implement an open method.")

    def close(self) -> None:
        """Closes the writer and finalizes logging."""
        raise NotImplementedError("Writers must implement a close method.")

    def write(self, result: OptimizeResult) -> None:
        """
        Logs the results of the current iteration.

        Args:
            result (OptimizeResult): The optimization results to log.
        """
        raise NotImplementedError("Writers must implement a write method.")

    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        raise NotImplementedError("Writers must implement a log_hyperparams method.")

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        """
        Logs plots of the model using provided plotting functions.

        Args:
            model (Module): The model to plot.
            iteration (int): The current iteration number.
            plotting_functions (tuple[PlottingFunction, ...]): Functions used to
                generate plots.
        """
        raise NotImplementedError("Writers must implement a plot method.")

    def log_model(
        self,
        model: Module,
        dataloader: Union[None, DataLoader, DictDataLoader],
    ) -> None:
        """
        Logs the model and associated data.

        Args:
            model (Module): The model to log.
            dataloader (DataLoader | DictDataLoader | None): DataLoader to use
                for model input.
        """
        raise NotImplementedError("Writers must implement a log_model method.")

    def print_metrics(self, result: OptimizeResult) -> None:
        """Prints the metrics and loss in a readable format.

        Args:
            result (OptimizeResult): The optimization results to display.
        """

        # Find the key in result.metrics that contains "loss" (case-insensitive)
        loss_key = next((k for k in result.metrics if "loss" in k.lower()), None)
        if loss_key:
            loss_value = result.metrics[loss_key]
            msg = f"Iteration {result.iteration: >7} | {loss_key.title()}: {loss_value:.7f} -"
        else:
            msg = f"Iteration {result.iteration: >7} | Loss: None -"
        msg += " ".join([f"{k}: {v:.7f}" for k, v in result.metrics.items() if k != loss_key])
        print(msg)


class TensorBoardWriter(BaseWriter):
    """Writer for logging to TensorBoard.

    Attributes:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
    """

    def __init__(self) -> None:
        self.writer = None

    def open(self, config: TrainConfig, iteration: int = None) -> SummaryWriter:
        """
        Opens the TensorBoard writer.

        Args:
            config: Configuration object containing settings for logging.
            iteration (int, optional): The iteration step to start logging from.
                Defaults to None.

        Returns:
            SummaryWriter: The initialized TensorBoard writer.
        """
        log_dir = str(config._log_folder)
        if isinstance(iteration, int):
            self.writer = SummaryWriter(log_dir=log_dir, purge_step=iteration)
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.close()

    def write(self, result: OptimizeResult) -> None:
        """
        Logs the results of the current iteration to TensorBoard.

        Args:
            result (OptimizeResult): The optimization results to log.
        """
        # Not writing loss as loss is available in the metrics
        # if result.loss is not None:
        #     self.writer.add_scalar("loss", float(result.loss), result.iteration)
        if self.writer:
            for key, value in result.metrics.items():
                self.writer.add_scalar(key, value, result.iteration)

    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters to TensorBoard.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        if self.writer:
            self.writer.add_hparams(hyperparams, {})

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        """
        Logs plots of the model using provided plotting functions.

        Args:
            model (Module): The model to plot.
            iteration (int): The current iteration number.
            plotting_functions (tuple[PlottingFunction, ...]): Functions used
                to generate plots.
        """
        if self.writer:
            for pf in plotting_functions:
                descr, fig = pf(model, iteration)
                self.writer.add_figure(descr, fig, global_step=iteration)

    def log_model(
        self,
        model: Module,
        dataloader: Union[None, DataLoader, DictDataLoader],
    ) -> None:
        """
        Logs the model.

        Currently not supported by TensorBoard.

        Args:
            model (Module): The model to log.
            dataloader (DataLoader | DictDataLoader | None): DataLoader to use
                for model input.
        """
        logger.warning("Model logging is not supported by tensorboard. No model will be logged.")


class MLFlowWriter(BaseWriter):
    """
    Writer for logging to MLflow.

    Attributes:
        run: The active MLflow run.
        mlflow: The MLflow module.
    """

    def __init__(self) -> None:
        self.run: Run
        self.mlflow: ModuleType

    def open(self, config: TrainConfig, iteration: int = None) -> ModuleType | None:
        """
        Opens the MLflow writer and initializes an MLflow run.

        Args:
            config: Configuration object containing settings for logging.
            iteration (int, optional): The iteration step to start logging from.
                Defaults to None.

        Returns:
            mlflow: The MLflow module instance.
        """
        import mlflow

        self.mlflow = mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", str(uuid4()))
        run_name = os.getenv("MLFLOW_RUN_NAME", str(uuid4()))

        if self.mlflow:
            self.mlflow.set_tracking_uri(tracking_uri)

            # Create or get the experiment
            exp_filter_string = f"name = '{experiment_name}'"
            experiments = self.mlflow.search_experiments(filter_string=exp_filter_string)
            if not experiments:
                self.mlflow.create_experiment(name=experiment_name)

            self.mlflow.set_experiment(experiment_name)
            self.run = self.mlflow.start_run(run_name=run_name, nested=False)

        return self.mlflow

    def close(self) -> None:
        """Closes the MLflow run."""
        if self.run:
            self.mlflow.end_run()

    def write(self, result: OptimizeResult) -> None:
        """
        Logs the results of the current iteration to MLflow.

        Args:
            result (OptimizeResult): The optimization results to log.
        """
        # Not writing loss as loss is available in the metrics
        # if result.loss is not None:
        #     self.mlflow.log_metric("loss", float(result.loss), step=result.iteration)
        if self.mlflow:
            self.mlflow.log_metrics(result.metrics, step=result.iteration)

    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters to MLflow.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        if self.mlflow:
            self.mlflow.log_params(hyperparams)

    def plot(
        self,
        model: Module,
        iteration: int,
        plotting_functions: tuple[PlottingFunction, ...],
    ) -> None:
        """
        Logs plots of the model using provided plotting functions.

        Args:
            model (Module): The model to plot.
            iteration (int): The current iteration number.
            plotting_functions (tuple[PlottingFunction, ...]): Functions used
                to generate plots.
        """
        if self.mlflow:
            for pf in plotting_functions:
                descr, fig = pf(model, iteration)
                self.mlflow.log_figure(fig, descr)

    def log_model(
        self,
        model: Module,
        dataloader: Union[None, DataLoader, DictDataLoader],
    ) -> None:
        """
        Logs the model and its signature to MLflow.

        Args:
            model (Module): The model to log.
            dataloader (DataLoader | DictDataLoader | None): DataLoader to
                use for model input.
        """
        if self.mlflow:
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
    """Factory method to get the appropriate writer based on the tracking tool.

    Args:
        tracking_tool (ExperimentTrackingTool): The experiment tracking tool to use.

    Returns:
        BaseWriter: An instance of the appropriate writer.
    """
    writer_class = WRITER_REGISTRY.get(tracking_tool)
    if writer_class:
        return writer_class()
    else:
        raise ValueError(f"Unsupported tracking tool: {tracking_tool}")
