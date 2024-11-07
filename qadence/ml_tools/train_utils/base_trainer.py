from __future__ import annotations

from contextlib import contextmanager
from logging import getLogger
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import nevergrad as ng
import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch import nn, optim
from torch.utils.data import DataLoader

from qadence.ml_tools.callbacks import CallbacksManager
from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import InfiniteTensorDataset
from qadence.ml_tools.loss import get_loss_fn
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.parameters import get_parameters

from .config_manager import ConfigManager

logger = getLogger(__name__)


class BaseTrainer:
    """Base class for training machine learning models using a given optimizer.

    The base class implementes contextmanagers for gradient based/free optimization,
    properties, property setters, input validations, callback decorator denerator,
    and empty hooks for different training steps.

    This class provides:
    - Context managers for enabling/disabling gradient-based optimization
    - Properties for managing models, optimizers, and dataloaders
    - Input validations and a callback decorator generator
    - Config and callback managers using the provided `TrainConfig`

    Attributes:
        use_grad (bool): Indicates if gradients are used for optimization. Default is True.

        _model (Optional[nn.Module]): The neural network model.
        _optimizer (Optional[Union[optim.Optimizer, NGOptimizer]]): The optimizer for training.
        _config (Optional[TrainConfig]): The configuration settings for training.
        _train_dataloader (Optional[DataLoader]): DataLoader for training data.
        _val_dataloader (Optional[DataLoader]): DataLoader for validation data.
        _test_dataloader (Optional[DataLoader]): DataLoader for testing data.

        optimize_step (Callable): Function for performing an optimization step.
        loss_fn (Callable): loss function to use.

        num_training_batches (int): Number of training batches.
        num_validation_batches (int): Number of validation batches.
        num_test_batches (int): Number of test batches.

        state (str): Current state in the training process
    """

    use_grad: bool = True

    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[optim.Optimizer, NGOptimizer, None],
        config: TrainConfig,
        loss_fn: Union[None, Callable, str],
        optimize_step: Callable = optimize_step,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        max_batches: int = None,
    ):
        """
        Initializes the BaseTrainer.

        Args:
            model ([nn.Module]): The model to train.
            optimizer (Optional[Union[optim.Optimizer, NGOptimizer, None]]): The optimizer
                for training.
            config ([TrainConfig]): The TrainConfig settings for training.
            loss_fn (Union[None, Callable, str]): The loss function to use.
                str input to be specified to use a default loss function.
                currently supported loss functions: 'mse', 'cross_entropy'
            train_dataloader (Optional[DataLoader]): DataLoader for training data.
                If the model does not need data to evaluvate loss, no dataset
                should be provided.
            val_dataloader (Optional[DataLoader]): DataLoader for validation data.
            test_dataloader (Optional[DataLoader]): DataLoader for testing data.
            max_batches (Optional[int]): Maximum number of batches to process per epoch.
                This is only valid in case of finite TensorDataset dataloaders.
                if max_batches is not None, the maximum number of batches used will
                be min(max_batches, len(dataloader.dataset))
                In case of InfiniteTensorDataset only 1 batch per eopch is used.
        """
        self._model: nn.Module
        self._optimizer: Union[optim.Optimizer, NGOptimizer, None]
        self._config: TrainConfig
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.max_batches = max_batches

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.num_training_batches = 1
        self.num_validation_batches = 1
        self.num_test_batches = 1

        self.loss_fn = get_loss_fn(loss_fn)
        self.optimize_step = optimize_step

        self.state = "idle"

    @property
    def model(self) -> nn.Module:
        """
        Returns the model if set, otherwise raises an error.

        Returns:
            nn.Module: The model.
        """
        if self._model is None:
            raise ValueError("Model has not been set.")
        return self._model

    @model.setter
    def model(self, model: nn.Module) -> None:
        """
        Sets the model, ensuring it is an instance of nn.Module.

        Args:
            model (Optional[nn.Module]): The neural network model.
        """
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of nn.Module or None.")
        self._model = model

    @property
    def optimizer(self) -> Union[optim.Optimizer, NGOptimizer, None]:
        """
        Returns the optimizer if set, otherwise raises an error.

        Returns:
            Union[optim.Optimizer, NGOptimizer]: The optimizer.
        """
        if self._optimizer is None:
            raise ValueError("Optimizer has not been set.")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[optim.Optimizer, NGOptimizer, None]) -> None:
        """
        Sets the optimizer, checking compatibility with gradient use.

        We also set up the budget/behaviour of different optimizers here.

        Args:
            optimizer (Union[optim.Optimizer, NGOptimizer]): The optimizer for training.
        """
        if optimizer is not None:
            if self.use_grad:
                if not isinstance(optimizer, optim.Optimizer):
                    raise TypeError("use_grad=True requires a PyTorch optimizer instance.")
            else:
                if not isinstance(optimizer, NGOptimizer):
                    raise TypeError("use_grad=False requires a Nevergrad optimizer instance.")
                else:
                    optimizer.budget = self.config.max_iter
                    optimizer.enable_pickling()
                    params = get_parameters(self.model).detach().numpy()
                    self.ng_params = ng.p.Array(init=params)

        self._optimizer = optimizer

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader, validating its type.

        Returns:
            DataLoader: The DataLoader for training data.
        """
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, dataloader: DataLoader) -> None:
        """
        Sets the training DataLoader and computes the number of batches.

        Args:
            dataloader (DataLoader): The DataLoader for training data.
        """
        self._validate_dataloader(dataloader, "train")
        self._train_dataloader = dataloader
        self.num_training_batches = self._compute_num_batches(dataloader)

    @property
    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader, validating its type.

        Returns:
            DataLoader: The DataLoader for validation data.
        """
        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, dataloader: DataLoader) -> None:
        """
        Sets the validation DataLoader and computes the number of batches.

        Args:
            dataloader (DataLoader): The DataLoader for validation data.
        """
        self._validate_dataloader(dataloader, "val")
        self._val_dataloader = dataloader
        self.num_validation_batches = self._compute_num_batches(dataloader)

    @property
    def test_dataloader(self) -> DataLoader:
        """
        Returns the test DataLoader, validating its type.

        Returns:
            DataLoader: The DataLoader for testing data.
        """
        return self._test_dataloader

    @test_dataloader.setter
    def test_dataloader(self, dataloader: DataLoader) -> None:
        """
        Sets the test DataLoader and computes the number of batches.

        Args:
            dataloader (DataLoader): The DataLoader for testing data.
        """
        self._validate_dataloader(dataloader, "test")
        self._test_dataloader = dataloader
        self.num_test_batches = self._compute_num_batches(dataloader)

    @property
    def config(self) -> TrainConfig:
        """
        Returns the training configuration.

        Returns:
            TrainConfig: The configuration object.
        """
        return self._config

    @config.setter
    def config(self, value: TrainConfig) -> None:
        """
        Sets the training configuration and initializes callback and config managers.

        Args:
            value (Optional[TrainConfig]): The configuration object.
        """
        if value and not isinstance(value, TrainConfig):
            raise TypeError("config must be an instance of TrainConfig.")
        self._config = value
        self.callback_manager = CallbacksManager(value)
        self.config_manager = ConfigManager(value)

    @classmethod
    def set_use_grad(cls, value: bool) -> None:
        """
        Sets the global use_grad flag.

        Args:
            value (bool): Whether to use gradient-based optimization.
        """
        if not isinstance(value, bool):
            raise TypeError("use_grad must be a boolean value.")
        cls.use_grad = value

    def _compute_num_batches(self, dataloader: DataLoader) -> int:
        """
        Computes the number of batches for the given DataLoader.

        Args:
            dataloader (DataLoader): The DataLoader for which to compute
                the number of batches.
        """
        if dataloader is None:
            return 0
        dataset = dataloader.dataset
        if isinstance(dataset, InfiniteTensorDataset):
            return 1
        else:
            return (
                min(self.max_batches, len(dataloader))
                if self.max_batches is not None
                else len(dataloader)
            )

    def _validate_dataloader(self, dataloader: DataLoader, dataloader_type: str) -> None:
        """
        Validates the type of the DataLoader and raises errors for unsupported types.

        Args:
            dataloader (DataLoader): The DataLoader to validate.
            dataloader_type (str): The type of DataLoader ("train", "val", or "test").
        """
        if dataloader is not None:
            if not isinstance(dataloader, DataLoader):
                raise NotImplementedError(
                    f"Unsupported dataloader type: {type(dataloader)}."
                    "The dataloader must be an instance of DataLoader."
                )
        if dataloader_type == "val" and self.config.val_every > 0:
            if not isinstance(dataloader, DataLoader):
                raise ValueError(
                    "If `config.val_every` is provided as an integer > 0, validation_dataloader"
                    "must be an instance of `DataLoader`."
                )

    @staticmethod
    def callback(phase: str) -> Callable:
        """
        Decorator for executing callbacks before and after a phase.

        Phase are different hooks during the training. List of valid
        phases is defined in Callbacks.
        We also update the current state of the training process in
        the callback decorator.

        Args:
            phase (str): The phase for which the callback is executed (e.g., "train",
            "train_epoch", "train_batch").

        Returns:
            Callable: The decorated function.
        """

        def decorator(method: Callable) -> Callable:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                start_event = f"on_{phase}_start"
                end_event = f"on_{phase}_end"

                self.state = start_event
                self.callback_manager.run_callbacks(trainer=self)
                result = method(self, *args, **kwargs)

                self.state = end_event
                # build_optimize_result method is defined in the trainer.
                self.build_optimize_result(result)
                self.callback_manager.run_callbacks(trainer=self)

                return result

            return wrapper

        return decorator

    @contextmanager
    def enable_grad_opt(self, optimizer: optim.Optimizer = None) -> Iterator[None]:
        """
        Context manager to temporarily enable gradient-based optimization.

        Args:
            optimizer Optional(optim.Optimizer): The PyTorch optimizer to use.
                If no optimizer is provided, default optimer for trainer
                object will be used.
        """
        original_mode = self.use_grad
        original_optimizer = self._optimizer
        try:
            self.use_grad = True
            self.callback_manager.use_grad = True
            self.optimizer = optimizer if optimizer else self.optimizer
            yield
        finally:
            self.use_grad = original_mode
            self.callback_manager.use_grad = original_mode
            self.optimizer = original_optimizer

    @contextmanager
    def disable_grad_opt(self, optimizer: NGOptimizer = None) -> Iterator[None]:
        """
        Context manager to temporarily disable gradient-based optimization.

        Args:
            optimizer Optional(NGOptimizer): The Nevergrad optimizer to use.
                If no optimizer is provided, default optimer for trainer
                object will be used.
        """
        original_mode = self.use_grad
        original_optimizer = self._optimizer
        try:
            self.use_grad = False
            self.callback_manager.use_grad = False
            self.optimizer = optimizer if optimizer else self.optimizer
            yield
        finally:
            self.use_grad = original_mode
            self.callback_manager.use_grad = original_mode
            self.optimizer = original_optimizer

    def on_train_start(self) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(
        self,
        train_losses: List[List[Tuple[torch.Tensor, Any]]],
        val_losses: Optional[List[List[Tuple[torch.Tensor, Any]]]] = None,
    ) -> None:
        """
        Called at the end of training.

        Args:
            train_losses: Metrics for the training losses.
                List    -> List                  -> Tuples
                Epochs  -> Training Batches      -> (loss, metrics)
            val_losses: Metrics for the validation losses.
                List    -> List                  -> Tuples
                Epochs  -> Validation Batches    -> (loss, metrics)
        """
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_train_epoch_end(self, train_epoch_loss_metrics: List[Tuple[torch.Tensor, Any]]) -> None:
        """
        Called at the end of each training epoch.

        Args:
            train_epoch_loss_metrics: Metrics for the training epoch losses.
                List                  -> Tuples
                Training Batches      -> (loss, metrics)
        """
        pass

    def on_val_epoch_start(self) -> None:
        """Called at the start of each validation epoch."""
        pass

    def on_val_epoch_end(self, val_epoch_loss_metrics: List[Tuple[torch.Tensor, Any]]) -> None:
        """
        Called at the end of each validation epoch.

        Args:
            val_epoch_loss_metrics: Metrics for the validation epoch loss.
                List                    -> Tuples
                Validation Batches      -> (loss, metrics)
        """
        pass

    def on_train_batch_start(self, batch: Tuple[torch.Tensor, ...] | None) -> None:
        """
        Called at the start of each training batch.

        Args:
            batch: A batch of data from the DataLoader. Typically a tuple containing
                input tensors and corresponding target tensors.
        """
        pass

    def on_train_batch_end(self, train_batch_loss_metrics: Tuple[torch.Tensor, Any]) -> None:
        """
        Called at the end of each training batch.

        Args:
            train_batch_loss_metrics: Metrics for the training batch loss.
                Tuple of (loss, metrics)
        """
        pass

    def on_val_batch_start(self, batch: Tuple[torch.Tensor, ...] | None) -> None:
        """
        Called at the start of each validation batch.

        Args:
            batch: A batch of data from the DataLoader. Typically a tuple containing
                input tensors and corresponding target tensors.
        """
        pass

    def on_val_batch_end(self, val_batch_loss_metrics: Tuple[torch.Tensor, Any]) -> None:
        """
        Called at the end of each validation batch.

        Args:
            val_batch_loss_metrics: Metrics for the validation batch loss.
                Tuple of (loss, metrics)
        """
        pass

    def on_test_batch_start(self, batch: Tuple[torch.Tensor, ...] | None) -> None:
        """
        Called at the start of each testing batch.

        Args:
            batch: A batch of data from the DataLoader. Typically a tuple containing
                input tensors and corresponding target tensors.
        """
        pass

    def on_test_batch_end(self, test_batch_loss_metrics: Tuple[torch.Tensor, Any]) -> None:
        """
        Called at the end of each testing batch.

        Args:
            test_batch_loss_metrics: Metrics for the testing batch loss.
                Tuple of (loss, metrics)
        """
        pass
