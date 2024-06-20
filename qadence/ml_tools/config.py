from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from matplotlib.figure import Figure
from torch.nn import Module

from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)


@dataclass
class TrainConfig:
    """Default config for the train function.

    The default value of
    each field can be customized with the constructor:

    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    c = TrainConfig(folder="/tmp/train")
    print(str(c)) # markdown-exec: hide
    ```
    """

    max_iter: int = 10000
    """Number of training iterations."""
    print_every: int = 1000
    """Print loss/metrics."""
    write_every: int = 50
    """Write tensorboard logs."""
    checkpoint_every: int = 5000
    """Write model/optimizer checkpoint."""
    plot_every: Optional[int] = None
    """Write figures.

    NOTE: currently only works with mlflow.
    """
    folder: Optional[Path] = None
    """Checkpoint/tensorboard logs folder."""
    create_subfolder_per_run: bool = False
    """Checkpoint/tensorboard logs stored in subfolder with name `<timestamp>_<PID>`.

    Prevents continuing from previous checkpoint, useful for fast prototyping.
    """
    checkpoint_best_only: bool = False
    """Write model/optimizer checkpoint only if a metric has improved."""
    validation_criterion: Optional[Callable] = None
    """A boolean function which evaluates a given validation metric is satisfied."""
    trainstop_criterion: Optional[Callable] = None
    """A boolean function which evaluates a given training stopping metric is satisfied."""
    batch_size: int = 1
    """The batch_size to use when passing a list/tuple of torch.Tensors."""
    verbose: bool = True
    """Whether or not to print out metrics values during training."""
    tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
    """The tracking tool of choice."""
    hyperparams: Optional[dict] = None
    """Hyperparameters to track."""
    plotting_functions: Optional[tuple[Callable[[Module, int], tuple[str, Figure]]]] = None
    """Functions for in-train plotting."""

    # mlflow_callbacks: list[Callable] = [write_mlflow_figure(), write_x()]

    def __post_init__(self) -> None:
        if self.folder:
            if isinstance(self.folder, str):  # type: ignore [unreachable]
                self.folder = Path(self.folder)  # type: ignore [unreachable]
            if self.create_subfolder_per_run:
                subfoldername = (
                    datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + hex(os.getpid())[2:]
                )
                self.folder = self.folder / subfoldername
        if self.trainstop_criterion is None:
            self.trainstop_criterion = lambda x: x <= self.max_iter
        if self.validation_criterion is None:
            self.validation_criterion = lambda x: False
        if self.plot_every and self.tracking_tool != ExperimentTrackingTool.MLFLOW:
            raise NotImplementedError("In-training plots are only available with mlflow tracking.")
        if self.plot_every and self.plotting_functions is None:
            logger.warning("Plots tracking is required, but no plotting functions are provided.")


@dataclass
class MLFlowConfig:
    """
    Example:

        export MLFLOW_TRACKING_URI=tracking_uri
        export MLFLOW_TRACKING_USERNAME=username
        export MLFLOW_TRACKING_PASSWORD=password
    """

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_TRACKING_USERNAME: str = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    MLFLOW_TRACKING_PASSWORD: str = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", str(uuid4()))
    RUN_NAME: str = os.getenv("MLFLOW_RUN_NAME", "test_0")

    def __post_init__(self) -> None:
        import mlflow

        if self.MLFLOW_TRACKING_USERNAME != "":
            logger.info(
                f"Intialized mlflow remote logging for user {self.MLFLOW_TRACKING_USERNAME}."
            )
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.EXPERIMENT)
        mlflow.start_run(run_name=self.RUN_NAME, nested=False)
