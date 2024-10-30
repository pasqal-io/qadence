from typing import Any, List, Union, Optional

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import OptimizeResult
from qadence.ml_tools.callback.callbacks import (
    Callback,
    PrintMetrics,
    WriteMetrics,
    PlotMetrics,
    LogHyperparameters,
    SaveCheckpoint,
    LoadCheckpoint,
    LogModelTracker,
    WriterSetup,
    WriterClose,
    SaveBestCheckpoint
)

from .writer_registry import BaseWriter

import logging

logger = logging.getLogger(__name__)


class CallbacksManager:
    use_grad: bool = True

    def __init__(self, config: TrainConfig = None):
        self.callbacks: List[Callback] = [
            self.get_callback_by_name(cb) if isinstance(cb, str) else cb
            for cb in (config.callbacks or [])
        ]
        self.config = config
        self.writer: Optional[BaseWriter] = None

    @classmethod
    def set_optimization_type(cls, use_grad: bool):
        if not isinstance(use_grad, bool):
            raise ValueError("use_grad must be a boolean value.")
        cls.use_grad = use_grad
        logger.debug(f"Optimization type set to {'use_grad' if use_grad else 'no_grad'}.")

    def initialize_callbacks(self):

        if self.config.verbose and self.config.print_every:
            self.add_callback("PrintMetrics", "on_train_epoch_end")
        if self.config.write_every:
            self.add_callback("WriteMetrics", "on_train_epoch_end")
        if self.config.plot_every:
            self.add_callback("PlotMetrics", "on_train_epoch_end")
        if self.config.checkpoint_every:
            self.add_callback("SaveCheckpoint", "on_train_epoch_end")

        if self.config.checkpoint_best_only:
            self.add_callback("SaveBestCheckpoint", "on_val_epoch_end")
        
        if self.config.hyperparams:
            self.add_callback("LogHyperparameters", "on_train_end")
        if self.config.log_model:
            self.add_callback("LogModelTracker", "on_train_end")

        self.add_callback("WriterClose", "on_train_end")
        

    def add_callback(self, callback: Union[Callback, str], on: Optional[str] = None):
        if isinstance(callback, str):
            callback_instance = self.get_callback_by_name(callback, on=on)
            if callback_instance:
                self.callbacks.append(callback_instance)
            else:
                logger.warning(f"Callback '{callback}' not recognized and will be skipped.")
        elif isinstance(callback, Callback):
            callback.on = on if on else callback.on
            self.callbacks.append(callback)
        else:
            logger.warning(f"Invalid callback type: {type(callback)}. Expected str or Callback instance.")

    def run_callbacks(
        self, 
        event: str, 
        trainer,  
    ) -> Any:
        return [
            callback(
                when=event,
                trainer=trainer,
                config=self.config,
                writer=self.writer
            )
            for callback in self.callbacks if callback.on == event
        ]

    def get_callback_by_name(self, name: str, on: Optional[str] = None) -> Optional[Callback]:
        if not on:
            on = "on_train_end"

        callback_map = {
            "PrintMetrics": PrintMetrics(on=on, called_every=self.config.print_every),
            "WriteMetrics": WriteMetrics(on=on, called_every=self.config.write_every),
            "PlotMetrics": PlotMetrics(on=on, called_every=self.config.plot_every),
            "SaveCheckpoint": SaveCheckpoint(on=on, called_every=self.config.checkpoint_every),
            "LoadCheckpoint": LoadCheckpoint(on=on, called_every=1),
            "LogModelTracker": LogModelTracker(on=on, called_every=1),
            "LogHyperparameters": LogHyperparameters(on=on, called_every=1),
            "WriterSetup": WriterSetup(on=on),
            "WriterClose": WriterClose(on=on),
        }

        if self.config.val_every:
            callback_map.update( { 
                "SaveBestCheckpoint": SaveBestCheckpoint(on=on, called_every=self.config.val_every)} )

        callback = callback_map.get(name)
        if not callback:
            logger.warning(f"Callback '{name}' not found in callback map.")
        return callback


    def start_training(self, trainer) -> None:
        self.initialize_callbacks()

        trainer.opt_result = OptimizeResult(trainer.global_step, trainer.model, trainer.optimizer)
        trainer.is_last_iteration = False

        load_checkpoint_callback = LoadCheckpoint(on="on_train_start", called_every=1)
        loaded_result = load_checkpoint_callback.run_callback(
            trainer=trainer,
            config=self.config,
            writer=None
        )

        if loaded_result:
            model, optimizer, init_iter = loaded_result
            if init_iter != 0:
                trainer.model = model
                trainer.optimizer = optimizer
                trainer.global_step = init_iter
                trainer.opt_result = OptimizeResult(init_iter, model, optimizer)
                logger.debug(f"Loaded model and optimizer from {self.config.log_folder}")

        writer_setup_callback = WriterSetup(on="on_train_start", called_every=1)
        self.writer = writer_setup_callback.run_callback(
            trainer=trainer,
            config=self.config,
            writer = self.writer
        )

    def end_training(
        self, 
        trainer,  
    ) -> None:
        trainer.is_last_iteration = True

        self.run_callbacks(
            event='on_train_end',
            trainer=trainer
        )
