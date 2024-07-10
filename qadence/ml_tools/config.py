from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from torch import Tensor

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
    plot_every: int = 5000
    """Write figures.

    NOTE: currently only works with mlflow.
    """
    folder: Path | None = None
    """Checkpoint/tensorboard logs folder."""
    create_subfolder_per_run: bool = False
    """Checkpoint/tensorboard logs stored in subfolder with name `<timestamp>_<PID>`.

    Prevents continuing from previous checkpoint, useful for fast prototyping.
    """
    checkpoint_best_only: bool = False
    """Write model/optimizer checkpoint only if a metric has improved."""
    validation_criterion: Callable | None = None
    """A boolean function which evaluates a given validation metric is satisfied."""
    trainstop_criterion: Optional[Callable] = None
    """A boolean function which evaluates a given training stopping metric is satisfied."""
    batch_size: int = 1
    """The batch_size to use when passing a list/tuple of torch.Tensors."""
    verbose: bool = True
    """Whether or not to print out metrics values during training."""
    tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
    """The tracking tool of choice."""
    hyperparams: dict = field(default_factory=dict)
    """Hyperparameters to track."""
    plotting_functions: tuple[Callable] = field(default_factory=tuple)  # type: ignore
    """Functions for in-train plotting."""

    # tensorboard only allows for certain types as hyperparameters
    _tb_allowed_hyperparams_types: tuple = field(
        default=(int, float, str, bool, Tensor), init=False, repr=False
    )

    def _filter_tb_hyperparams(self) -> None:
        keys_to_remove = [
            key
            for key, value in self.hyperparams.items()
            if not isinstance(value, TrainConfig._tb_allowed_hyperparams_types)
        ]
        if keys_to_remove:
            logger.warning(
                f"Tensorboard cannot log the following hyperparameters: {keys_to_remove}."
            )
            for key in keys_to_remove:
                self.hyperparams.pop(key)

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
        if self.hyperparams and self.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
            self._filter_tb_hyperparams()
        if self.plotting_functions and self.tracking_tool != ExperimentTrackingTool.MLFLOW:
            logger.warning("In-training plots are only available with mlflow tracking.")
        if not self.plotting_functions and self.tracking_tool == ExperimentTrackingTool.MLFLOW:
            logger.warning("Tracking with mlflow, but no plotting functions provided.")


@dataclass
class MLFlowConfig:
    """
    Configuration for mlflow tracking.

    Example:

        export MLFLOW_TRACKING_URI=tracking_uri
        export MLFLOW_TRACKING_USERNAME=username
        export MLFLOW_TRACKING_PASSWORD=password
    """

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    """The URI of the mlflow tracking server.

    An empty string, or a local file path, prefixed with file:/.
    Data is stored locally at the provided file (or ./mlruns if empty).
    """

    MLFLOW_TRACKING_USERNAME: str = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    """The username for the mlflow tracking server."""

    MLFLOW_TRACKING_PASSWORD: str = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    """The password for the mlflow tracking server."""

    EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", str(uuid4()))
    """The name of the experiment.

    If None or empty, a new experiment is created with a random UUID.
    """

    RUN_NAME: str = os.getenv("MLFLOW_RUN_NAME", "test_0")
    """The name of the run."""

    def __post_init__(self) -> None:
        import mlflow

        if self.MLFLOW_TRACKING_USERNAME != "":
            logger.info(
                f"Intialized mlflow remote logging for user {self.MLFLOW_TRACKING_USERNAME}."
            )
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        # activate existing or create experiment
        exp_filter_string = f"name = '{self.EXPERIMENT}'"
        if not mlflow.search_experiments(filter_string=exp_filter_string):
            mlflow.create_experiment(name=self.EXPERIMENT)

        mlflow.set_experiment(self.EXPERIMENT)
        mlflow.start_run(run_name=self.RUN_NAME, nested=False)
