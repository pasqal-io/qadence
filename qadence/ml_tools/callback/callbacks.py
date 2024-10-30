from typing import Callable, Optional, Any, Union

from qadence.ml_tools.callback.saveload import load_checkpoint, write_checkpoint
from qadence.ml_tools.callback.writer_registry import get_writer  

# Define callback types
CallbackFunction = Callable[[Any], Any]
CallbackConditionFunction = Callable[[Any], bool]


class Callback:
    VALID_ON_VALUES = [
        "on_train_start", "on_train_end",

        "on_train_epoch_start", "on_train_epoch_end",
        "on_train_batch_start", "on_train_batch_end",

        "on_val_epoch_start", "on_val_epoch_end",
        "on_val_batch_start", "on_val_batch_end",

        "on_test_batch_start", "on_test_batch_end"
    ]

    def __init__(
        self,
        on: str = "on_train_end",
        called_every: int = 1,
        callback: Optional[CallbackFunction] = None,
        callback_condition: Optional[CallbackConditionFunction] = None,
        modify_optimize_result: Optional[Union[CallbackFunction, dict[str, Any]]] = None
    ):
        if called_every <= 0:
            raise ValueError("called_every must be strictly positive.")
        if on not in self.VALID_ON_VALUES:
            raise ValueError(f"Invalid value for 'on'. Must be one of {self.VALID_ON_VALUES}.")

        self.callback = callback
        self.on = on
        self.called_every = called_every 
        self.callback_condition = callback_condition or (lambda _: True)

        if isinstance(modify_optimize_result, dict):
            self.modify_optimize_result = (
                lambda opt_res: opt_res.extra.update(modify_optimize_result) or opt_res
            )
        else:
            self.modify_optimize_result = modify_optimize_result or (lambda opt_res: opt_res)

    def __call__(self, when: str, trainer: Any, config: Any, writer: Any) -> Any:
        opt_result = trainer.opt_result
        if self.on == when:
            if opt_result:
                opt_result = self.modify_optimize_result(opt_result)
            if (
                (trainer.opt_result.iteration % self.called_every == 0 
                 and self.callback_condition(opt_result)) 
                 or when == "on_train_end"
                ):
                return self.run_callback(trainer, config, writer)

    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        if self.callback is not None:
            return self.callback(trainer, config, writer)
        raise NotImplementedError("Subclasses should override the run_callback method.")


class PrintMetrics(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        opt_result = trainer.opt_result
        writer.print_metrics(opt_result.loss, opt_result.metrics, opt_result.iteration)


class WriteMetrics(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        opt_result = trainer.opt_result
        writer.write(opt_result.loss, opt_result.metrics, opt_result.iteration)


class PlotMetrics(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        opt_result = trainer.opt_result
        plotting_functions = config.plotting_functions
        writer.plot(trainer.model, opt_result.iteration, plotting_functions)


class LogHyperparameters(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        hyperparams = config.hyperparams
        metrics = trainer.opt_result.metrics
        writer.log_hyperparams(hyperparams, metrics)


class SaveCheckpoint(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        folder = config.log_folder
        model = trainer.model
        optimizer = trainer.optimizer
        opt_result = trainer.opt_result
        write_checkpoint(folder, model, optimizer, opt_result.iteration)


class SaveBestCheckpoint(SaveCheckpoint):
    def __init__(self, on: str, called_every: int):
        super().__init__(on=on, called_every=called_every)
        self.best_loss = float('inf')

    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        opt_result = trainer.opt_result
        if config.validation_criterion(opt_result.loss, self.best_loss, config.val_epsilon):
            self.best_loss = opt_result.loss
            super().run_callback(trainer, config, writer)


class LoadCheckpoint(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        folder = config.log_folder
        model = trainer.model
        optimizer = trainer.optimizer
        device = trainer.log_device
        # model_ckpt_name = trainer.model_ckpt_name if trainer.model_ckpt_name else ""
        # opt_ckpt_name = trainer.opt_ckpt_name if trainer.opt_ckpt_name else ""
        return load_checkpoint(folder, model, optimizer, device=device)

class LogModelTracker(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        model = trainer.model
        dataloader = trainer.train_dataloader
        writer.log_model(model, dataloader)


class WriterSetup(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        tracking_tool = config.tracking_tool
        writer = get_writer(tracking_tool)
        writer.open(config, iteration=trainer.global_step)
        return writer


class WriterClose(Callback):
    def run_callback(self, trainer: Any, config: Any, writer: Any) -> Any:
        if writer:
            writer.close()
