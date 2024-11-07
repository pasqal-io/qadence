from __future__ import annotations

import os
import datetime
import math
from pathlib import Path
from logging import getLogger
from typing import Union, Any
from dataclasses import field

from torch import Tensor

from qadence.types import ExperimentTrackingTool
from qadence.ml_tools.config import TrainConfig

logger = getLogger(__name__)

class ConfigManager:
    """A class to manage and initialize the configuration for a 
    machine learning training run using TrainConfig.

    Attributes:
        config (TrainConfig): The training configuration object 
        containing parameters and settings.
    """

    optimization_type: str = 'with_grad'

    def __init__(self, config: TrainConfig):
        """
        Initialize the ConfigManager with a given training configuration.

        Args:
            config (TrainConfig): The training configuration object.
        """
        self.config: TrainConfig = config

    def initialize_config(self) -> None:
        """
        Initialize the configuration by setting up the folder structure,
        handling hyperparameters, deriving additional parameters, 
        and logging warnings.
        """
        self._initialize_folder()
        self._handle_hyperparams()
        self._derive_parameters()
        self._log_warnings()

    def _initialize_folder(self) -> None:
        """
        Initialize the folder structure for logging. Creates a log folder
        if the folder path is specified in the configuration.
        config has three parameters
        - folder: The root folder for logging
        - subfolders: list of subfolders inside `folder` that are used for logging 
        - log_folder: folder currently used for loggin.
        """
        if self.config.folder:
            self.config._log_folder = self._create_log_folder(self.config.folder)

    def _create_log_folder(self, root_folder: Union[str, Path]) -> Path:
        """
        Create a log folder in the specified root folder, adding subfolders if required.

        Args:
            root_folder (Union[str, Path]): The root folder where the log folder will be created.

        Returns:
            Path: The path to the created log folder.
        """
        root_folder_path = Path(root_folder)
        root_folder_path.mkdir(parents=True, exist_ok=True)

        if self.config.create_subfolder_per_run:
            self._add_subfolder()
            log_folder = root_folder_path / self.config._subfolders[-1]
        else:
            if len(self.config._subfolders) == 0:
                self._add_subfolder()
            log_folder = root_folder_path / self.config._subfolders[-1]

        log_folder.mkdir(parents=True, exist_ok=True)
        return Path(log_folder)

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

    def _handle_hyperparams(self) -> None:
        """
        Handle and filter hyperparameters based on the selected tracking tool.
        Removes incompatible hyperparameters when using TensorBoard. 
        """
        # tensorboard only allows for certain types as hyperparameters
    
        if self.config.hyperparams and self.config.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
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
            logger.warning(f"Tensorboard cannot log the following hyperparameters: {keys_to_remove}.")
            for key in keys_to_remove:
                self.config.hyperparams.pop(key)

    def _derive_parameters(self) -> None:
        """
        Derive additional parameters for the training configuration.
        Sets the stopping criterion if it is not already defined.
        """
        if self.config.trainstop_criterion is None:
            self.config.trainstop_criterion = lambda x: x <= self.config.max_iter

    def _log_warnings(self) -> None:
        """
        Log warnings for incompatible configurations related to tracking tools and plotting functions.
        """
        if self.config.plotting_functions and self.config.tracking_tool != ExperimentTrackingTool.MLFLOW:
            logger.warning("In-training plots are only available with mlflow tracking.")
        if not self.config.plotting_functions and self.config.tracking_tool == ExperimentTrackingTool.MLFLOW:
            logger.warning("Tracking with mlflow, but no plotting functions provided.")
