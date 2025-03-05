from __future__ import annotations

import datetime
import os
from logging import getLogger
from pathlib import Path

from torch import Tensor

from qadence.ml_tools.config import TrainConfig
from qadence.types import ExperimentTrackingTool

logger = getLogger("ml_tools")


class ConfigManager:
    """A class to manage and initialize the configuration for a.

    machine learning training run using TrainConfig.

    Attributes:
        config (TrainConfig): The training configuration object
        containing parameters and settings.
    """

    optimization_type: str = "with_grad"

    def __init__(self, config: TrainConfig):
        """
        Initialize the ConfigManager with a given training configuration.

        Args:
            config (TrainConfig): The training configuration object.
        """
        self.config: TrainConfig = config

    def initialize_config(self) -> None:
        """
        Initialize the configuration by setting up the folder structure,.

        handling hyperparameters, deriving additional parameters,
        and logging warnings.
        """
        self._log_warnings()
        self._initialize_folder()
        self._handle_hyperparams()
        self._setup_additional_configuration()

    def _initialize_folder(self) -> None:
        """
        Initialize the folder structure for logging.

        Creates a log folder
        if the folder path is specified in the configuration.
        config has three parameters
        - folder: The root folder for logging
        - subfolders: list of subfolders inside `folder` that are used for logging
        - log_folder: folder currently used for logging.
        """
        self.config.log_folder = self._createlog_folder(self.config.root_folder)

    def _createlog_folder(self, root_folder: str | Path) -> Path:
        """
        Create a log folder in the specified root folder, adding subfolders if required.

        Args:
            root_folder (str | Path): The root folder where the log folder will be created.

        Returns:
            Path: The path to the created log folder.
        """
        self._added_new_subfolder: bool = False
        root_folder_path = Path(root_folder)
        root_folder_path.mkdir(parents=True, exist_ok=True)

        if self.config.create_subfolder_per_run:
            self._add_subfolder()
            log_folder = root_folder_path / self.config._subfolders[-1]
        else:
            if self.config._subfolders:
                # self.config.log_folder is an old subfolder.
                log_folder = Path(self.config.log_folder)
            else:
                if self.config.log_folder == Path("./"):
                    # A subfolder must be created (no specific folder given to config).
                    self._add_subfolder()
                    log_folder = root_folder_path / self.config._subfolders[-1]
                else:
                    # The folder is one and fully specified by the user.
                    log_folder = Path(self.config.log_folder)

        log_folder.mkdir(parents=True, exist_ok=True)
        return log_folder

    def _add_subfolder(self) -> None:
        """
        Add a unique subfolder name to the configuration for logging.

        The subfolder name includes a run ID, timestamp, and process ID in hexadecimal format.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        pid_hex = hex(os.getpid())[2:]
        run_id = len(self.config._subfolders) + 1
        subfolder_name = f"{run_id}_{timestamp}_{pid_hex}"
        self.config._subfolders.append(str(subfolder_name))
        self._added_new_subfolder = True

    def _handle_hyperparams(self) -> None:
        """
        Handle and filter hyperparameters based on the selected tracking tool.

        Removes incompatible hyperparameters when using TensorBoard.
        """
        # tensorboard only allows for certain types as hyperparameters

        if (
            self.config.hyperparams
            and self.config.tracking_tool == ExperimentTrackingTool.TENSORBOARD
        ):
            self._filter_tb_hyperparams()

    def _filter_tb_hyperparams(self) -> None:
        """
        Filter out hyperparameters that cannot be logged by TensorBoard.

        Logs a warning for the removed hyperparameters.
        """

        # tensorboard only allows for certain types as hyperparameters
        tb_allowed_hyperparams_types: tuple = (int, float, str, bool, Tensor)
        keys_to_remove = [
            key
            for key, value in self.config.hyperparams.items()
            if not isinstance(value, tb_allowed_hyperparams_types)
        ]
        if keys_to_remove:
            logger.warning(
                f"Tensorboard cannot log the following hyperparameters: {keys_to_remove}."
            )
            for key in keys_to_remove:
                self.config.hyperparams.pop(key)

    def _setup_additional_configuration(self) -> None:
        """
        Derive additional parameters for the training configuration.

        Sets the stopping criterion if it is not already defined.
        """
        if self.config.trainstop_criterion is None:
            return

    def _log_warnings(self) -> None:
        """
        Log warnings for incompatible configurations related to tracking tools.

        and plotting functions.
        """
        if (
            self.config.plotting_functions
            and self.config.tracking_tool != ExperimentTrackingTool.MLFLOW
        ):
            logger.warning("In-training plots are only available with mlflow tracking.")
        if (
            not self.config.plotting_functions
            and self.config.tracking_tool == ExperimentTrackingTool.MLFLOW
        ):
            logger.warning("Tracking with mlflow, but no plotting functions provided.")
        if self.config.plot_every and not self.config.plotting_functions:
            logger.warning(
                "`plot_every` is only available when `plotting_functions` are provided."
                "No plots will be saved."
            )
        if self.config.checkpoint_best_only and not self.config.validation_criterion:
            logger.warning(
                "`Checkpoint_best_only` is only available when `validation_criterion` is provided."
                "No checkpoints will be saved."
            )
        if self.config.log_folder != Path("./") and self.config.root_folder != Path("./qml_logs"):
            logger.warning("Both `log_folder` and `root_folder` provided by the user.")
        if self.config.log_folder != Path("./") and self.config.create_subfolder_per_run:
            logger.warning(
                "`log_folder` is invalid when `create_subfolder_per_run` = True."
                "`root_folder` (default qml_logs) will be used to save logs."
            )
