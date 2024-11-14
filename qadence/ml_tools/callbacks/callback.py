from __future__ import annotations

from typing import Any, Callable

from qadence.ml_tools.callbacks.saveload import load_checkpoint, write_checkpoint
from qadence.ml_tools.callbacks.writer_registry import BaseWriter
from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import OptimizeResult
from qadence.ml_tools.stages import TrainingStage

# Define callback types
CallbackFunction = Callable[..., Any]
CallbackConditionFunction = Callable[..., bool]


class Callback:
    """Base class for defining various training callbacks.

    Attributes:
        on (str): The event on which to trigger the callback.
            Must be a valid on value from: ["train_start", "train_end",
                "train_epoch_start", "train_epoch_end", "train_batch_start",
                "train_batch_end","val_epoch_start", "val_epoch_end",
                "val_batch_start", "val_batch_end", "test_batch_start",
                "test_batch_end"]
        called_every (int): Frequency of callback calls in terms of iterations.
        callback (CallbackFunction | None): The function to call if the condition is met.
        callback_condition (CallbackConditionFunction | None): Condition to check before calling.
        modify_optimize_result (CallbackFunction | dict[str, Any] | None):
            Function to modify `OptimizeResult`.

    A callback can be defined in two ways:

    1. **By providing a callback function directly in the base class**:
       This is useful for simple callbacks that don't require subclassing.

       Example:
       ```python exec="on" source="material-block" result="json"
       from qadence.ml_tools.callbacks import Callback

       def custom_callback_function(trainer, config, writer):
           print("Custom callback executed.")

       custom_callback = Callback(
           on="train_end",
           called_every=5,
           callback=custom_callback_function
       )
       ```

    2. **By inheriting and implementing the `run_callback` method**:
       This is suitable for more complex callbacks that require customization.

       Example:
       ```python exec="on" source="material-block" result="json"
       from qadence.ml_tools.callbacks import Callback
       class CustomCallback(Callback):
           def run_callback(self, trainer, config, writer):
               print("Custom behavior in the inherited run_callback method.")

       custom_callback = CustomCallback(on="train_end", called_every=10)
       ```
    """

    VALID_ON_VALUES = [
        "train_start",
        "train_end",
        "train_epoch_start",
        "train_epoch_end",
        "train_batch_start",
        "train_batch_end",
        "val_epoch_start",
        "val_epoch_end",
        "val_batch_start",
        "val_batch_end",
        "test_batch_start",
        "test_batch_end",
    ]

    def __init__(
        self,
        on: str | TrainingStage = "idle",
        called_every: int = 1,
        callback: CallbackFunction | None = None,
        callback_condition: CallbackConditionFunction | None = None,
        modify_optimize_result: CallbackFunction | dict[str, Any] | None = None,
    ):
        if not isinstance(called_every, int):
            raise ValueError("called_every must be a positive integer or 0")

        self.callback: CallbackFunction | None = callback
        self.on: str | TrainingStage = on
        self.called_every: int = called_every
        self.callback_condition = callback_condition or (lambda _: True)

        if isinstance(modify_optimize_result, dict):
            self.modify_optimize_result = (
                lambda opt_res: opt_res.extra.update(modify_optimize_result) or opt_res
            )
        else:
            self.modify_optimize_result = modify_optimize_result or (lambda opt_res: opt_res)

    @property
    def on(self) -> TrainingStage | str:
        """
        Returns the TrainingStage.

        Returns:
            TrainingStage: TrainingStage for the callback
        """
        return self._on

    @on.setter
    def on(self, on: str | TrainingStage) -> None:
        """
        Sets the training stage on for the callback.

        Args:
            on (str | TrainingStage): TrainingStage for the callback
        """
        if isinstance(on, str):
            if on not in self.VALID_ON_VALUES:
                raise ValueError(f"Invalid value for 'on'. Must be one of {self.VALID_ON_VALUES}.")
            self._on = TrainingStage(on)
        elif isinstance(on, TrainingStage):
            self._on = on
        else:
            raise ValueError("Invalid value for 'on'. Must be `str` or `TrainingStage`.")

    def _should_call(self, when: str, opt_result: OptimizeResult) -> bool:
        """Checks if the callback should be called.

        Args:
            when (str): The event when the callback is considered for execution.
            opt_result (OptimizeResult): The current optimization results.

        Returns:
            bool: Whether the callback should be called.
        """
        if when in [TrainingStage("train_start"), TrainingStage("train_end")]:
            return True
        if self.called_every == 0 or opt_result.iteration == 0:
            return False
        if opt_result.iteration % self.called_every == 0 and self.callback_condition(opt_result):
            return True
        return False

    def __call__(
        self, when: TrainingStage, trainer: Any, config: TrainConfig, writer: BaseWriter
    ) -> Any:
        """Executes the callback if conditions are met.

        Args:
            when (str): The event when the callback is triggered.
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: Result of the callback function if executed.
        """
        opt_result = trainer.opt_result
        if self.on == when:
            if opt_result:
                opt_result = self.modify_optimize_result(opt_result)
            if self._should_call(when, opt_result):
                return self.run_callback(trainer, config, writer)

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Executes the defined callback.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: Result of the callback execution.

        Raises:
            NotImplementedError: If not implemented in subclasses.
        """
        if self.callback is not None:
            return self.callback(trainer, config, writer)
        raise NotImplementedError("Subclasses should override the run_callback method.")


class PrintMetrics(Callback):
    """Callback to print metrics using the writer.

    The `PrintMetrics` callback can be added to the `TrainConfig`
    callbacks as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `PrintMetrics`, include it in the `callbacks` list when
    setting up your `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.callbacks import PrintMetrics

    # Create an instance of the PrintMetrics callback
    print_metrics_callback = PrintMetrics(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[print_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Prints metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        opt_result = trainer.opt_result
        writer.print_metrics(opt_result)


class WriteMetrics(Callback):
    """Callback to write metrics using the writer.

    The `WriteMetrics` callback can be added to the `TrainConfig` callbacks as
    a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `WriteMetrics`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.callbacks import WriteMetrics

    # Create an instance of the WriteMetrics callback
    write_metrics_callback = WriteMetrics(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[write_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Writes metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        opt_result = trainer.opt_result
        writer.write(opt_result)


class PlotMetrics(Callback):
    """Callback to plot metrics using the writer.

    The `PlotMetrics` callback can be added to the `TrainConfig` callbacks as
    a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `PlotMetrics`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.callbacks import PlotMetrics

    # Create an instance of the PlotMetrics callback
    plot_metrics_callback = PlotMetrics(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[plot_metrics_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Plots metrics using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        opt_result = trainer.opt_result
        plotting_functions = config.plotting_functions
        writer.plot(trainer.model, opt_result.iteration, plotting_functions)


class LogHyperparameters(Callback):
    """Callback to log hyperparameters using the writer.

    The `LogHyperparameters` callback can be added to the `TrainConfig` callbacks
    as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `LogHyperparameters`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.callbacks import LogHyperparameters

    # Create an instance of the LogHyperparameters callback
    log_hyper_callback = LogHyperparameters(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[log_hyper_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Logs hyperparameters using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        hyperparams = config.hyperparams
        writer.log_hyperparams(hyperparams)


class SaveCheckpoint(Callback):
    """Callback to save a model checkpoint.

    The `SaveCheckpoint` callback can be added to the `TrainConfig` callbacks
    as a custom user defined callback.

    Example Usage in `TrainConfig`:
    To use `SaveCheckpoint`, include it in the `callbacks` list when setting up your
    `TrainConfig`:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.callbacks import SaveCheckpoint

    # Create an instance of the SaveCheckpoint callback
    save_checkpoint_callback = SaveCheckpoint(on = "val_batch_end", called_every = 100)

    config = TrainConfig(
        max_iter=10000,
        # Print metrics every 1000 training epochs
        print_every=1000,
        # Add the custom callback that runs every 100 val_batch_end
        callbacks=[save_checkpoint_callback]
    )
    ```
    """

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Saves a model checkpoint.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        folder = config.log_folder
        model = trainer.model
        optimizer = trainer.optimizer
        opt_result = trainer.opt_result
        write_checkpoint(folder, model, optimizer, opt_result.iteration)


class SaveBestCheckpoint(SaveCheckpoint):
    """Callback to save the best model checkpoint based on a validation criterion."""

    def __init__(self, on: str, called_every: int):
        """Initializes the SaveBestCheckpoint callback.

        Args:
            on (str): The event to trigger the callback.
            called_every (int): Frequency of callback calls in terms of iterations.
        """
        super().__init__(on=on, called_every=called_every)
        self.best_loss = float("inf")

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Saves the checkpoint if the current loss is better than the best loss.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        opt_result = trainer.opt_result
        if config.validation_criterion and config.validation_criterion(
            opt_result.loss, self.best_loss, config.val_epsilon
        ):
            self.best_loss = opt_result.loss

            folder = config.log_folder
            model = trainer.model
            optimizer = trainer.optimizer
            opt_result = trainer.opt_result
            write_checkpoint(folder, model, optimizer, "best")


class LoadCheckpoint(Callback):
    """Callback to load a model checkpoint."""

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Loads a model checkpoint.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.

        Returns:
            Any: The result of loading the checkpoint.
        """
        folder = config.log_folder
        model = trainer.model
        optimizer = trainer.optimizer
        device = trainer.log_device
        return load_checkpoint(folder, model, optimizer, device=device)


class LogModelTracker(Callback):
    """Callback to log the model using the writer."""

    def run_callback(self, trainer: Any, config: TrainConfig, writer: BaseWriter) -> Any:
        """Logs the model using the writer.

        Args:
            trainer (Any): The training object.
            config (TrainConfig): The configuration object.
            writer (BaseWriter ): The writer object for logging.
        """
        model = trainer.model
        writer.log_model(
            model, trainer.train_dataloader, trainer.val_dataloader, trainer.test_dataloader
        )
