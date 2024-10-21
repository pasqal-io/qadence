from __future__ import annotations

from typing import Callable, Any, Optional, List, Union

from qadence.ml_tools.data import OptimizeResult
from qadence.ml_tools.printing import (
    plot_tracker,
    print_metrics,
    write_tracker,
)
from qadence.ml_tools.saveload import write_checkpoint
from qadence.types import ExperimentTrackingTool


CallbackFunction = Callable[[OptimizeResult], None]
CallbackConditionFunction = Callable[[OptimizeResult], bool]

class Callback:
    """Callback functions are calling in train functions.

    Each callback function should take at least as first input
    an OptimizeResult instance.

    Note: when setting call_after_opt to True, we skip
    verifying iteration % called_every == 0.

    Attributes:
        callback (CallbackFunction): Callback function accepting an
            OptimizeResult as first argument.
        callback_condition (CallbackConditionFunction | None, optional): Function that
            conditions the call to callback. Defaults to None.
        modify_optimize_result (CallbackFunction | dict[str, Any] | None, optional):
            Function that modify the OptimizeResult before callback.
            For instance, one can change the `extra` (dict) argument to be used in callback.
            If a dict is provided, the `extra` field of OptimizeResult is updated with the dict.
        called_every (int, optional): Callback to be called each `called_every` epoch.
            Defaults to 1.
            If callback_condition is None, we set
            callback_condition to returns True when iteration % called_every == 0.
    """
    
    def __init__(
        self,
        callback: CallbackFunction,
        callback_condition: Optional[CallbackConditionFunction] = None,
        modify_optimize_result: Optional[Union[CallbackFunction, dict[str, Any]]] = None,
        called_every: int = 1,
    ):
        """Initialized Callback.

        Args:
            callback (CallbackFunction): Callback function accepting an
                OptimizeResult as ifrst argument.
            callback_condition (CallbackConditionFunction | None, optional): Function that
                conditions the call to callback. Defaults to None.
            modify_optimize_result (CallbackFunction | dict[str, Any] | None , optional):
                Function that modify the OptimizeResult before callback. If a dict
                is provided, this updates the `extra` field of OptimizeResult.
            called_every (int, optional): Callback to be called each `called_every` epoch.
                Defaults to 1.
                If callback_condition is None, we set
                callback_condition to returns True when iteration % called_every == 0.
        """
        if called_every <= 0:
            raise ValueError("called_every must be strictly positive.")

        self.callback = callback
        self.called_every = called_every
        self.callback_condition = callback_condition or (lambda _: True)

        if isinstance(modify_optimize_result, dict):
            self.modify_optimize_result = lambda opt_res: opt_res.extra.update(modify_optimize_result) or opt_res
        else:
            self.modify_optimize_result = modify_optimize_result or (lambda opt_res: opt_res)

    def __call__(self, opt_result: OptimizeResult, is_last_iteration: bool = False) -> Any:
        """Apply callback if conditions are met.

        Note that the current result may be modified by specifying a function
        `modify_optimize_result` for instance to add inputs to the `extra` argument
        of the current OptimizeResult.

        Args:
            opt_result (OptimizeResult): Current result.
            is_last_iteration (bool, optional): When True,
                avoid verifying modulo. Defaults to False.
                Useful when call_after_opt is True.

        Returns:
            Any: The result of the callback.
        """
        opt_result = self.modify_optimize_result(opt_result)
        if opt_result.iteration % self.called_every == 0 and self.callback_condition(opt_result):
            return self.callback(opt_result)
        if is_last_iteration and self.callback_condition(opt_result):
            return self.callback(opt_result)


class CallbacksManager:
    """
    Manages and triggers callbacks during specific training events.

    The `CallbacksManager` is responsible for registering and executing callbacks at different stages 
    of the training loop (e.g., at the start or end of an epoch, or after a batch is processed).
    It also supports setting the optimization type to handle different behaviors depending on whether 
    gradients are used.

    Attributes:
        writer: A writer object (e.g., TensorBoard, logger) to log training metrics.
        config: A configuration object containing settings such as callback intervals and verbosity.
        callbacks (dict): A dictionary mapping events (e.g., 'train_start', 'epoch_end') to lists of callbacks.
        optimization_type (str): Class-level attribute to specify the optimization type ('with_grad' or 'without_grad').
    """

    optimization_type = 'with_grad'  # Default class-level optimization type
    
    def __init__(self, writer, config):
        """
        Initialize the CallbacksManager and set up default callbacks.

        Args:
            writer: A writer object (e.g., TensorBoard, logger) to log training metrics.
            config: A configuration object containing settings such as callback intervals and verbosity.
        """
        self.writer = writer
        self.config = config
        self.callbacks = {event: [] for event in ['train_start', 'train_end', 'epoch_start', 'epoch_end', 'batch_start', 'batch_end']}
        self._initialize_default_callbacks()

    @classmethod
    def set_optimization_type(cls, opt_type: str):
        """
        Set the optimization type (either 'with_grad' or 'without_grad') for the manager.

        This method changes the behavior of the manager based on whether gradients are used during optimization.

        Args:
            opt_type (str): The optimization type to set, either 'with_grad' or 'without_grad'.
        """
        if opt_type not in ['with_grad', 'without_grad']:
            raise ValueError("Optimization type must be either 'with_grad' or 'without_grad'.")
        cls.optimization_type = opt_type

    def _initialize_default_callbacks(self):
        """
        Set up default callbacks based on the configuration and optimization type.

        This method sets up basic callbacks for different stages of training. Depending on whether the 
        optimization uses gradients or not, certain callbacks (e.g., for checkpointing) may be included or excluded.
        """
        self.callbacks = {event: [] for event in ['train_start', 'train_end', 'epoch_start', 'epoch_end', 'batch_start', 'batch_end']}
        if self.config.verbose:
            self.add_callback("print", "batch_end")
        
        # If optimization type is 'with_grad', include more callbacks
        if CallbacksManager.optimization_type == 'with_grad':
            if self.config.write_every > 0:
                self.add_callback("write", "epoch_end")
            if self.config.plot_every > 0:
                self.add_callback("plot", "epoch_end")
            if self.config.checkpoint_every > 0:
                self.add_callback("checkpoint", "epoch_end")
        elif CallbacksManager.optimization_type == 'without_grad':
            # If without gradients, restrict certain callbacks
            if self.config.write_every > 0:
                self.add_callback("write", "epoch_end")

    def add_callback(self, callback: Union[Callback, str], event: Union[str, List[str]]):
        """
        Register a callback for one or more events.

        This method allows a callback (either a `Callback` object or the name of a predefined callback)
        to be registered for one or more training events (e.g., 'batch_end', 'epoch_end').

        Args:
            callback (Union[Callback, str]): The callback to register, or the name of a predefined callback.
            event (Union[str, List[str]]): One or more events during which the callback should be triggered.
        """
        if isinstance(callback, str):
            callback = self._get_callback_by_name(callback)
        
        events = [event] if isinstance(event, str) else event
        for evt in events:
            if evt not in self.callbacks:
                raise ValueError(f"Unsupported event: {evt}")
            self.callbacks[evt].append(callback)

    def _get_callback_by_name(self, name: str) -> Callback:
        """
        Return a pre-configured callback based on a name.

        Args:
            name (str): The name of the callback to return ('print', 'write', 'plot', 'checkpoint').

        Returns:
            Callback: The callback corresponding to the given name.
        """
        callback_map = {
            "print": self._create_print_callback,
            "write": self._create_write_callback,
            "plot": self._create_plot_callback,
            "checkpoint": self._create_checkpoint_callback
        }
        if name not in callback_map:
            raise ValueError(f"Unknown callback name: {name}")
        return callback_map[name](self.config, self.writer)

    def run(self, event: str, opt_res: OptimizeResult, is_last_iteration: bool = False):
        """
        Run callbacks for the specified event.

        This method triggers all callbacks that are registered for a particular event (e.g., 'batch_end').

        Args:
            event (str): The event that occurred (e.g., 'batch_end', 'epoch_end').
            opt_res (OptimizeResult): The current state of optimization, passed to each callback.
            is_last_iteration (bool, optional): Whether this is the final iteration of training. Defaults to False.
        """
        if event not in self.callbacks:
            raise ValueError(f"Unsupported event: {event}")
        self._run_callbacks(self.callbacks[event], opt_res, is_last_iteration)

    def _run_callbacks(self, callbacks: List[Callback], opt_res: OptimizeResult, is_last_iteration: bool = False) -> None:
        """Run a list of Callback given the current OptimizeResult.

        Used in train functions.

        Args:
            callback_iterable (list[Callback]): Iterable of Callbacks
            opt_res (OptimizeResult): Current optimization result,
            is_last_iteration (bool, optional): Whether we reached the last iteration or not.
                Defaults to False.
        """
        for callback in callbacks:
            callback(opt_res, is_last_iteration)
    
    # Static methods to create specific callbacks
    @staticmethod
    def create_print_callback(config, writer) -> Callback:
        """
        Create and return the print callback.

        Args:
            config: The configuration object, which contains `print_every` setting.
            writer: A writer object to which the metrics are logged.

        Returns:
            Callback: A callback that prints metrics at the end of each batch.
        """
        return Callback(
            callback=lambda opt_res: print_metrics(opt_res.loss, opt_res.metrics, opt_res.iteration),
            called_every=config.print_every
        )

    @staticmethod
    def create_write_callback(config, writer) -> Callback:
        """
        Create and return the write callback.

        Args:
            config: The configuration object, which contains `write_every` setting.
            writer: A writer object to log metrics.

        Returns:
            Callback: A callback that writes metrics at the end of each epoch.
        """
        return Callback(
            callback=lambda opt_res: write_tracker(writer, opt_res.loss, opt_res.metrics, opt_res.iteration),
            called_every=config.write_every
        )

    @staticmethod
    def create_plot_callback(config, writer) -> Callback:
        """
        Create and return the plot callback.

        Args:
            config: The configuration object, which contains `plot_every` and `plotting_functions`.
            writer: A writer object to log plots.

        Returns:
            Callback: A callback that generates plots at the end of each epoch.
        """
        return Callback(
            callback=lambda opt_res: plot_tracker(writer, opt_res.model, opt_res.iteration, config.plotting_functions),
            called_every=config.plot_every
        )

    @staticmethod
    def create_checkpoint_callback(config, writer) -> Callback:
        """
        Create and return the checkpoint callback.

        Args:
            config: The configuration object, which contains `checkpoint_every` and `folder`.
            writer: Unused, but kept for consistency with other callbacks.

        Returns:
            Callback: A callback that saves checkpoints at the end of each epoch.
        """
        return Callback(
            callback=lambda opt_res: write_checkpoint(config.folder, opt_res.model, opt_res.optimizer, opt_res.iteration),
            called_every=config.checkpoint_every
        )