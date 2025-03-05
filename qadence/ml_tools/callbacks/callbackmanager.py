from __future__ import annotations

import copy
import logging
from typing import Any

from qadence.ml_tools.callbacks.callback import (
    Callback,
    LoadCheckpoint,
    LogHyperparameters,
    LogModelTracker,
    PlotMetrics,
    PrintMetrics,
    SaveBestCheckpoint,
    SaveCheckpoint,
    WriteMetrics,
)
from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import OptimizeResult
from qadence.ml_tools.stages import TrainingStage

from .writer_registry import get_writer

logger = logging.getLogger("ml_tools")


class CallbacksManager:
    """Manages and orchestrates the execution of various training callbacks.

    Provides the start training and end training methods.

    Attributes:
        use_grad (bool): Indicates whether to use gradients in callbacks.
        config (TrainConfig): The training configuration object.
        callbacks (List[Callback]): List of callback instances to be executed.
        writer (Optional[BaseWriter]): The writer instance for logging metrics and information.
    """

    use_grad: bool = True

    callback_map = {
        "PrintMetrics": PrintMetrics,
        "WriteMetrics": WriteMetrics,
        "PlotMetrics": PlotMetrics,
        "SaveCheckpoint": SaveCheckpoint,
        "LoadCheckpoint": LoadCheckpoint,
        "LogModelTracker": LogModelTracker,
        "LogHyperparameters": LogHyperparameters,
        "SaveBestCheckpoint": SaveBestCheckpoint,
    }

    def __init__(self, config: TrainConfig):
        """
        Initializes the CallbacksManager with a training configuration.

        Args:
            config (TrainConfig): The training configuration object.
        """
        self.config = config
        tracking_tool = self.config.tracking_tool
        self.writer = get_writer(tracking_tool)
        self.callbacks: list[Callback] = []

    @classmethod
    def set_use_grad(cls, use_grad: bool) -> None:
        """
        Sets whether gradients should be used in callbacks.

        Args:
            use_grad (bool): A boolean indicating whether to use gradients.
        """
        if not isinstance(use_grad, bool):
            raise ValueError("use_grad must be a boolean value.")
        cls.use_grad = use_grad

    def initialize_callbacks(self) -> None:
        """Initializes and adds the necessary callbacks based on the configuration."""
        # Train Start
        self.callbacks = copy.deepcopy(self.config.callbacks)
        self.add_callback("PlotMetrics", "train_start")
        if self.config.val_every:
            self.add_callback("WriteMetrics", "train_start")
            # only save the first checkpoint if not checkpoint_best_only
            if not self.config.checkpoint_best_only:
                self.add_callback("SaveCheckpoint", "train_start")

        # Checkpointing
        if self.config.checkpoint_best_only:
            self.add_callback("SaveBestCheckpoint", "val_epoch_end", self.config.val_every)
        elif self.config.checkpoint_every:
            self.add_callback("SaveCheckpoint", "train_epoch_end", self.config.checkpoint_every)

        # Printing
        if self.config.verbose and self.config.print_every:
            self.add_callback("PrintMetrics", "train_epoch_end", self.config.print_every)

        # Plotting
        if self.config.plot_every:
            self.add_callback("PlotMetrics", "train_epoch_end", self.config.plot_every)

        # Writing
        if self.config.write_every:
            self.add_callback("WriteMetrics", "train_epoch_end", self.config.write_every)
        if self.config.val_every:
            self.add_callback("WriteMetrics", "val_epoch_end", self.config.val_every)

        # Train End
        # Hyperparameters
        if self.config.hyperparams:
            self.add_callback("LogHyperparameters", "train_end")
        # Log model
        if self.config.log_model:
            self.add_callback("LogModelTracker", "train_end")
        if self.config.plot_every:
            self.add_callback("PlotMetrics", "train_end")
        # only save the last checkpoint if not checkpoint_best_only
        if not self.config.checkpoint_best_only:
            self.add_callback("SaveCheckpoint", "train_end")
        self.add_callback("WriteMetrics", "train_end")

    def add_callback(
        self, callback: str | Callback, on: str | TrainingStage, called_every: int = 1
    ) -> None:
        """
        Adds a callback to the manager.

        Args:
            callback (str | Callback): The callback instance or name.
            on (str | TrainingStage): The event on which to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
        """
        if isinstance(callback, str):
            callback_class = self.callback_map.get(callback)
            if callback_class:
                # Create an instance of the callback class
                callback_instance = callback_class(on=on, called_every=called_every)
                self.callbacks.append(callback_instance)
            else:
                logger.warning(f"Callback '{callback}' not recognized and will be skipped.")
        elif isinstance(callback, Callback):
            callback.on = on
            callback.called_every = called_every
            self.callbacks.append(callback)
        else:
            logger.warning(
                f"Invalid callback type: {type(callback)}. Expected str or Callback instance."
            )

    def run_callbacks(self, trainer: Any) -> Any:
        """
        Runs callbacks that match the current training state.

        Args:
            trainer (Any): The training object managing the training process.

        Returns:
            Any: Results of the executed callbacks.
        """
        return [
            callback(
                when=trainer.training_stage, trainer=trainer, config=self.config, writer=self.writer
            )
            for callback in self.callbacks
            if callback.on == trainer.training_stage
        ]

    def start_training(self, trainer: Any) -> None:
        """
        Initializes callbacks and starts the training process.

        Args:
            trainer (Any): The training object managing the training process.
        """
        # Clear all handlers from the logger
        self.initialize_callbacks()

        trainer.opt_result = OptimizeResult(trainer.global_step, trainer.model, trainer.optimizer)
        trainer.is_last_iteration = False

        # Load checkpoint only if a new subfolder was NOT recently added
        if not trainer.config_manager._added_new_subfolder:
            load_checkpoint_callback = LoadCheckpoint(on="train_start", called_every=1)
            loaded_result = load_checkpoint_callback.run_callback(
                trainer=trainer,
                config=self.config,
                writer=None,  # type: ignore[arg-type]
            )

            if loaded_result:
                model, optimizer, init_iter = loaded_result
                if isinstance(init_iter, (int, str)):
                    trainer.model = model
                    trainer.optimizer = optimizer
                    trainer.global_step = (
                        init_iter if isinstance(init_iter, int) else trainer.global_step
                    )
                    trainer.current_epoch = (
                        init_iter if isinstance(init_iter, int) else trainer.current_epoch
                    )
                    trainer.opt_result = OptimizeResult(trainer.current_epoch, model, optimizer)
                    logger.debug(f"Loaded model and optimizer from {self.config.log_folder}")

        # Setup writer
        if trainer.accelerator.rank == 0:
            self.writer.open(self.config, iteration=trainer.global_step)

    def end_training(self, trainer: Any) -> None:
        """
        Cleans up and finalizes the training process.

        Args:
            trainer (Any): The training object managing the training process.
        """
        if trainer.accelerator.rank == 0 and self.writer:
            self.writer.close()
