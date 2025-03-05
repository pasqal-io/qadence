from __future__ import annotations

import os
from abc import ABC, abstractmethod
from logging import getLogger
from types import ModuleType
from typing import Any, Callable, Union
from uuid import uuid4

from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, OptimizeResult
from qadence.types import ExperimentTrackingTool

logger = getLogger("ml_tools")

# Type aliases
PlottingFunction = Callable[[Module, int], tuple[str, Figure]]
InputData = Union[Tensor, dict[str, Tensor]]


class BaseWriter(ABC):
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

    run: Any  # [attr-defined]

    @abstractmethod
    def open(self, config: TrainConfig, iteration: int | None = None) -> Any:
        """
        Opens the writer and prepares it for logging.

        Args:
            config: Configuration object containing settings for logging.
            iteration (int, optional): The iteration step to start logging from.
                Defaults to None.
        """
        raise NotImplementedError("Writers must implement an open method.")

    @abstractmethod
    def close(self) -> None:
        """Closes the writer and finalizes logging."""
        raise NotImplementedError("Writers must implement a close method.")

    @abstractmethod
    def write(self, iteration: int, metrics: dict) -> None:
        """
        Logs the results of the current iteration.

        Args:
            iteration (int): The current training iteration.
            metrics (dict): A dictionary of metrics to log, where keys are metric names
                            and values are the corresponding metric values.
        """
        raise NotImplementedError("Writers must implement a write method.")

    @abstractmethod
    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        raise NotImplementedError("Writers must implement a log_hyperparams method.")

    @abstractmethod
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

    @abstractmethod
    def log_model(
        self,
        model: Module,
        train_dataloader: DataLoader | DictDataLoader | None = None,
        val_dataloader: DataLoader | DictDataLoader | None = None,
        test_dataloader: DataLoader | DictDataLoader | None = None,
    ) -> None:
        """
        Logs the model and associated data.

        Args:
            model (Module): The model to log.
            train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
            val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.
            test_dataloader (DataLoader | DictDataLoader |  None): DataLoader for testing data.
        """
        raise NotImplementedError("Writers must implement a log_model method.")

    def print_metrics(self, result: OptimizeResult) -> None:
        """Prints the metrics and loss in a readable format.

        Args:
            result (OptimizeResult): The optimization results to display.
        """

        # Find the key in result.metrics that contains "loss" (case-insensitive)
        loss_key = next((k for k in result.metrics if "loss" in k.lower()), None)
        initial = f"P {result.rank: >2}|{result.device: <7}| Iteration {result.iteration: >7}| "
        if loss_key:
            loss_value = result.metrics[loss_key]
            msg = initial + f"{loss_key.title()}: {loss_value:.7f} -"
        else:
            msg = initial + f"Loss: None -"
        msg += " ".join([f"{k}: {v:.7f}" for k, v in result.metrics.items() if k != loss_key])
        print(msg)


class TensorBoardWriter(BaseWriter):
    """Writer for logging to TensorBoard.

    Attributes:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
    """

    def __init__(self) -> None:
        self.writer = None

    def open(self, config: TrainConfig, iteration: int | None = None) -> SummaryWriter:
        """
        Opens the TensorBoard writer.

        Args:
            config: Configuration object containing settings for logging.
            iteration (int, optional): The iteration step to start logging from.
                Defaults to None.

        Returns:
            SummaryWriter: The initialized TensorBoard writer.
        """
        log_dir = str(config.log_folder)
        purge_step = iteration if isinstance(iteration, int) else None
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)
        return self.writer

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.close()

    def write(self, iteration: int, metrics: dict) -> None:
        """
        Logs the results of the current iteration to TensorBoard.

        Args:
            iteration (int): The current training iteration.
            metrics (dict): A dictionary of metrics to log, where keys are metric names
                            and values are the corresponding metric values.
        """
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, iteration)
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing."
            )

    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters to TensorBoard.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        if self.writer:
            self.writer.add_hparams(hyperparams, {})
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing"
            )

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
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing"
            )

    def log_model(
        self,
        model: Module,
        train_dataloader: DataLoader | DictDataLoader | None = None,
        val_dataloader: DataLoader | DictDataLoader | None = None,
        test_dataloader: DataLoader | DictDataLoader | None = None,
    ) -> None:
        """
        Logs the model.

        Currently not supported by TensorBoard.

        Args:
            model (Module): The model to log.
            train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
            val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.
            test_dataloader (DataLoader | DictDataLoader |  None): DataLoader for testing data.
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
        try:
            from mlflow.entities import Run
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Please install qadence with the mlflow feature: "
                "`pip install qadence[mlflow]`."
            )

        self.run: Run
        self.mlflow: ModuleType

    def open(self, config: TrainConfig, iteration: int | None = None) -> ModuleType | None:
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

    def write(self, iteration: int, metrics: dict) -> None:
        """
        Logs the results of the current iteration to MLflow.

        Args:
            iteration (int): The current training iteration.
            metrics (dict): A dictionary of metrics to log, where keys are metric names
                            and values are the corresponding metric values.
        """
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=iteration)
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing."
            )

    def log_hyperparams(self, hyperparams: dict) -> None:
        """
        Logs hyperparameters to MLflow.

        Args:
            hyperparams (dict): A dictionary of hyperparameters to log.
        """
        if self.mlflow:
            self.mlflow.log_params(hyperparams)
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing"
            )

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
        else:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing"
            )

    def get_signature_from_dataloader(
        self, model: Module, dataloader: DataLoader | DictDataLoader | None
    ) -> Any:
        """
        Infers the signature of the model based on the input data from the dataloader.

        Args:
            model (Module): The model to use for inference.
            dataloader (DataLoader | DictDataLoader |  None): DataLoader for model inputs.

        Returns:
            Optional[Any]: The inferred signature, if available.
        """
        from mlflow.models import infer_signature

        if dataloader is None:
            return None

        xs: InputData
        xs, *_ = next(iter(dataloader))
        preds = model(xs)

        if isinstance(xs, Tensor):
            xs = xs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            return infer_signature(xs, preds)

        return None

    def log_model(
        self,
        model: Module,
        train_dataloader: DataLoader | DictDataLoader | None = None,
        val_dataloader: DataLoader | DictDataLoader | None = None,
        test_dataloader: DataLoader | DictDataLoader | None = None,
    ) -> None:
        """
        Logs the model and its signature to MLflow using the provided data loaders.

        Args:
            model (Module): The model to log.
            train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
            val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.
            test_dataloader (DataLoader | DictDataLoader |  None): DataLoader for testing data.
        """
        if not self.mlflow:
            raise RuntimeError(
                "The writer is not initialized."
                "Please call the 'writer.open()' method before writing"
            )

        signatures = self.get_signature_from_dataloader(model, train_dataloader)
        self.mlflow.pytorch.log_model(model, artifact_path="model", signature=signatures)


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
