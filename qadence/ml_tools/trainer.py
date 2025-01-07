from __future__ import annotations

import copy
from itertools import islice
from logging import getLogger
from typing import Any, Callable, Iterable, cast

import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import complex128, float32, float64, nn, optim
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.utils.data import DataLoader

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, OptimizeResult
from qadence.ml_tools.optimize_step import optimize_step, update_ng_parameters
from qadence.ml_tools.stages import TrainingStage

from .train_utils.base_trainer import BaseTrainer

logger = getLogger("ml_tools")


class Trainer(BaseTrainer):
    """Trainer class to manage and execute training, validation, and testing loops for a model (eg.

    QNN).

    This class handles the overall training process, including:
    - Managing epochs and steps
    - Handling data loading and batching
    - Computing and updating gradients
    - Logging and monitoring training metrics

    Attributes:
        current_epoch (int): The current epoch number.
        global_step (int): The global step across all epochs.
        log_device (str): Device for logging, default is "cpu".
        device (torch_device): Device used for computation.
        dtype (torch_dtype | None): Data type used for computation.
        data_dtype (torch_dtype | None): Data type for data.
            Depends on the model's data type.

    Inherited Attributes:
        use_grad (bool): Indicates if gradients are used for optimization. Default is True.

        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer | NGOptimizer | None): The optimizer for training.
        config (TrainConfig): The configuration settings for training.
        train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
        val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.
        test_dataloader (DataLoader | DictDataLoader |  None): DataLoader for testing data.

        optimize_step (Callable): Function for performing an optimization step.
        loss_fn (Callable): loss function to use.

        num_training_batches (int): Number of training batches.
        num_validation_batches (int): Number of validation batches.
        num_test_batches (int): Number of test batches.

        state (str): Current state in the training process

    Default training routine
    ```
    for epoch in max_iter + 1:
        # Training
        for batch in train_batches:
            train model
        # Validation
        if val_every % epoch == 0:
            for batch in val_batches:
                train model
    ```

    Notes:
        - In case of InfiniteTensorDataset, number of batches = 1.
        - In case of TensorDataset, number of batches are default.
        - Training is run for max_iter + 1 epochs. Epoch 0 logs untrained model.
        - Please look at the CallbackManager initialize_callbacks method to review the default
            logging behavior.

    Examples:

    ```python
    import torch
    from torch.optim import SGD
    from qadence import (
        feature_map,
        hamiltonian_factory,
        hea,
        QNN,
        QuantumCircuit,
        TrainConfig,
        Z,
    )
    from qadence.ml_tools.trainer import Trainer
    from qadence.ml_tools.optimize_step import optimize_step
    from qadence.ml_tools import TrainConfig
    from qadence.ml_tools.data import to_dataloader

    # Initialize the model
    n_qubits = 2
    fm = feature_map(n_qubits)
    ansatz = hea(n_qubits=n_qubits, depth=2)
    observable = hamiltonian_factory(n_qubits, detuning=Z)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")

    # Set up the optimizer
    optimizer = SGD(model.parameters(), lr=0.001)

    # Use TrainConfig for configuring the training process
    config = TrainConfig(
        max_iter=100,
        print_every=10,
        write_every=10,
        checkpoint_every=10,
        val_every=10
    )

    # Create the Trainer instance with TrainConfig
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        loss_fn="mse",
        optimize_step=optimize_step
    )

    batch_size = 25
    x = torch.linspace(0, 1, 32).reshape(-1, 1)
    y = torch.sin(x)
    train_loader = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    val_loader = to_dataloader(x, y, batch_size=batch_size, infinite=False)

    # Train the model
    model, optimizer = trainer.fit(train_loader, val_loader)
    ```

    This also supports both gradient based and gradient free optimization.
    The default support is for gradient based optimization.

    Notes:

    - **set_use_grad()** (*class level*):This method is used to set the global `use_grad` flag,
        controlling whether the trainer uses gradient-based optimization.
    ```python
    # gradient based
    Trainer.set_use_grad(True)

    # gradient free
    Trainer.set_use_grad(False)
    ```
    - **Context Managers** (*instance level*):  `enable_grad_opt()` and `disable_grad_opt()` are
        context managers that temporarily switch the optimization mode for specific code blocks.
        This is useful when you want to mix gradient-based and gradient-free optimization
        in the same training process.
    ```python
    # gradient based
    with trainer.enable_grad_opt(optimizer):
        trainer.fit()

    # gradient free
    with trainer.disable_grad_opt(ng_optimizer):
        trainer.fit()
    ```

    Examples

    *Gradient based optimization example Usage*:
    ```python
    from torch import optim
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    Trainer.set_use_grad(True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        loss_fn="mse"
    )
    trainer.fit(train_loader, val_loader)
    ```
    or
    ```python
    trainer = Trainer(
        model=model,
        config=config,
        loss_fn="mse"
    )
    with trainer.enable_grad_opt(optimizer):
        trainer.fit(train_loader, val_loader)
    ```

    *Gradient free optimization example Usage*:
    ```python
    import nevergrad as ng
    from qadence.ml_tools.parameters import num_parameters
    ng_optimizer = ng.optimizers.NGOpt(
                    budget=config.max_iter, parametrization= num_parameters(model)
                    )

    Trainer.set_use_grad(False)
    trainer = Trainer(
        model=model,
        optimizer=ng_optimizer,
        config=config,
        loss_fn="mse"
    )
    trainer.fit(train_loader, val_loader)
    ```
    or
    ```python
    import nevergrad as ng
    from qadence.ml_tools.parameters import num_parameters
    ng_optimizer = ng.optimizers.NGOpt(
            budget=config.max_iter, parametrization= num_parameters(model)
            )

    trainer = Trainer(
        model=model,
        config=config,
        loss_fn="mse"
    )
    with trainer.disable_grad_opt(ng_optimizer):
        trainer.fit(train_loader, val_loader)
    ```
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | NGOptimizer | None,
        config: TrainConfig,
        loss_fn: str | Callable = "mse",
        train_dataloader: DataLoader | DictDataLoader | None = None,
        val_dataloader: DataLoader | DictDataLoader | None = None,
        test_dataloader: DataLoader | DictDataLoader | None = None,
        optimize_step: Callable = optimize_step,
        device: torch_device | None = None,
        dtype: torch_dtype | None = None,
        max_batches: int | None = None,
    ):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (optim.Optimizer | NGOptimizer | None): The optimizer for training.
            config (TrainConfig): Training configuration object.
            loss_fn (str | Callable ): Loss function used for training.
                If not specified, default mse loss will be used.
            train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
            val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.
            test_dataloader (DataLoader | DictDataLoader |  None): DataLoader for test data.
            optimize_step (Callable): Function to execute an optimization step.
            device (torch_device): Device to use for computation.
            dtype (torch_dtype): Data type for computation.
            max_batches (int | None): Maximum number of batches to process per epoch.
                This is only valid in case of finite TensorDataset dataloaders.
                if max_batches is not None, the maximum number of batches used will
                be min(max_batches, len(dataloader.dataset))
                In case of InfiniteTensorDataset only 1 batch per epoch is used.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            loss_fn=loss_fn,
            optimize_step=optimize_step,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            max_batches=max_batches,
        )
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.log_device: str = "cpu" if device is None else device
        self.device: torch_device | None = device
        self.dtype: torch_dtype | None = dtype
        self.data_dtype: torch_dtype | None = None
        self.stop_training: bool = False
        if self.dtype:
            self.data_dtype = float64 if (self.dtype == complex128) else float32

    def fit(
        self,
        train_dataloader: DataLoader | DictDataLoader | None = None,
        val_dataloader: DataLoader | DictDataLoader | None = None,
    ) -> tuple[nn.Module, optim.Optimizer]:
        """
        Fits the model using the specified training configuration.

        The dataloaders can be provided to train on new datasets, or the default dataloaders
        provided in the trainer will be used.

        Args:
            train_dataloader (DataLoader | DictDataLoader |  None): DataLoader for training data.
            val_dataloader (DataLoader | DictDataLoader |  None): DataLoader for validation data.

        Returns:
            tuple[nn.Module, optim.Optimizer]: The trained model and optimizer.
        """
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
        if val_dataloader is not None:
            self.val_dataloader = val_dataloader

        self._fit_setup()
        self._train()
        self._fit_end()
        self.training_stage = TrainingStage("idle")
        return self.model, self.optimizer

    def _fit_setup(self) -> None:
        """
        Sets up the training environment, initializes configurations,.

        and moves the model to the specified device and data type.
        The callback_manager.start_training takes care of loading checkpoint,
        and setting up the writer.
        """
        self.stop_training = False
        self.config_manager.initialize_config()
        self.callback_manager.start_training(trainer=self)

        # Move model to device
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module.to(device=self.device, dtype=self.dtype)
        else:
            self.model = self.model.to(device=self.device, dtype=self.dtype)

        # Progress bar for training visualization
        self.progress: Progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        )

        # Quick Fix for iteration 0
        self._reset_model_and_opt()

        # Run validation at the start if specified in the configuration
        self.perform_val = self.config.val_every > 0
        if self.perform_val:
            self.run_validation(self.val_dataloader)

    def _fit_end(self) -> None:
        """Finalizes the training and closes the writer."""
        self.callback_manager.end_training(trainer=self)

    @BaseTrainer.callback("train")
    def _train(self) -> list[list[tuple[torch.Tensor, dict[str, Any]]]]:
        """
        Runs the main training loop, iterating over epochs.

        Returns:
            list[list[tuple[torch.Tensor, dict[str, Any]]]]: Training loss
            metrics for all epochs.
                list    -> list                  -> tuples
                Epochs  -> Training Batches      -> (loss, metrics)
        """
        self.on_train_start()
        train_losses = []
        val_losses = []

        with self.progress:
            train_task = self.progress.add_task(
                "Training", total=self.config_manager.config.max_iter
            )
            if self.perform_val:
                val_task = self.progress.add_task(
                    "Validation",
                    total=(self.config_manager.config.max_iter + 1) / self.config.val_every,
                )
            for epoch in range(
                self.global_step, self.global_step + self.config_manager.config.max_iter + 1
            ):
                if not self.stop_training:
                    try:
                        self.current_epoch = epoch
                        self.on_train_epoch_start()
                        train_epoch_loss_metrics = self.run_training(self.train_dataloader)
                        train_losses.append(train_epoch_loss_metrics)
                        self.on_train_epoch_end(train_epoch_loss_metrics)

                        # Run validation periodically if specified
                        if self.perform_val and self.current_epoch % self.config.val_every == 0:
                            self.on_val_epoch_start()
                            val_epoch_loss_metrics = self.run_validation(self.val_dataloader)
                            val_losses.append(val_epoch_loss_metrics)
                            self.on_val_epoch_end(val_epoch_loss_metrics)
                            self.progress.update(val_task, advance=1)

                        self.progress.update(train_task, advance=1)
                    except KeyboardInterrupt:
                        logger.info("Terminating training gracefully after the current iteration.")
                        break

        self.on_train_end(train_losses, val_losses)
        return train_losses

    @BaseTrainer.callback("train_epoch")
    def run_training(self, dataloader: DataLoader) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """
        Runs the training for a single epoch, iterating over multiple batches.

        Args:
            dataloader (DataLoader): DataLoader for training data.

        Returns:
            list[tuple[torch.Tensor, dict[str, Any]]]: Loss and metrics for each batch.
                list                  -> tuples
                Training Batches      -> (loss, metrics)
        """
        self.model.train()
        train_epoch_loss_metrics = []
        # Quick Fix for iteration 0
        self._reset_model_and_opt()

        for batch in self._batch_iter(dataloader, self.num_training_batches):
            self.on_train_batch_start(batch)
            train_batch_loss_metrics = self.run_train_batch(batch)
            train_epoch_loss_metrics.append(train_batch_loss_metrics)
            self.on_train_batch_end(train_batch_loss_metrics)

        return train_epoch_loss_metrics

    @BaseTrainer.callback("train_batch")
    def run_train_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Runs a single training batch, performing optimization.

        We use the step function to optimize the model based on use_grad.
            use_grad = True entails gradient based optimization, for which we use
            optimize_step function.
            use_grad = False entails gradient free optimization, for which we use
            update_ng_parameters function.

        Args:
            batch (tuple[torch.Tensor, ...]): Batch of data from the DataLoader.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Loss and metrics for the batch.
                tuple of (loss, metrics)
        """

        if self.use_grad:
            # Perform gradient-based optimization
            loss_metrics = self.optimize_step(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                xs=batch,
                device=self.device,
                dtype=self.data_dtype,
            )
        else:
            # Perform optimization using Nevergrad
            loss, metrics, ng_params = update_ng_parameters(
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                data=batch,
                ng_params=self.ng_params,  # type: ignore[arg-type]
            )
            self.ng_params = ng_params
            loss_metrics = loss, metrics

        return self._modify_batch_end_loss_metrics(loss_metrics)

    @BaseTrainer.callback("val_epoch")
    def run_validation(self, dataloader: DataLoader) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """
        Runs the validation loop for a single epoch, iterating over multiple batches.

        Args:
            dataloader (DataLoader): DataLoader for validation data.

        Returns:
            list[tuple[torch.Tensor, dict[str, Any]]]: Loss and metrics for each batch.
                list                  -> tuples
                Validation Batches      -> (loss, metrics)
        """
        self.model.eval()
        val_epoch_loss_metrics = []

        for batch in self._batch_iter(dataloader, self.num_validation_batches):
            self.on_val_batch_start(batch)
            val_batch_loss_metrics = self.run_val_batch(batch)
            val_epoch_loss_metrics.append(val_batch_loss_metrics)
            self.on_val_batch_end(val_batch_loss_metrics)

        return val_epoch_loss_metrics

    @BaseTrainer.callback("val_batch")
    def run_val_batch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Runs a single validation batch.

        Args:
            batch (tuple[torch.Tensor, ...]): Batch of data from the DataLoader.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Loss and metrics for the batch.
        """
        with torch.no_grad():
            loss_metrics = self.loss_fn(self.model, batch)
        return self._modify_batch_end_loss_metrics(loss_metrics)

    def test(self, test_dataloader: DataLoader = None) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        """
        Runs the testing loop if a test DataLoader is provided.

        if the test_dataloader is not provided, default test_dataloader defined
        in the Trainer class is used.

        Args:
            test_dataloader (DataLoader): DataLoader for test data.

        Returns:
            list[tuple[torch.Tensor, dict[str, Any]]]: Loss and metrics for each batch.
                list                    -> tuples
                Test Batches            -> (loss, metrics)
        """
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

        self.model.eval()
        test_loss_metrics = []

        for batch in self._batch_iter(test_dataloader, self.num_training_batches):
            self.on_test_batch_start(batch)
            loss_metrics = self.run_test_batch(batch)
            test_loss_metrics.append(loss_metrics)
            self.on_test_batch_end(loss_metrics)

        return test_loss_metrics

    @BaseTrainer.callback("test_batch")
    def run_test_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Runs a single test batch.

        Args:
            batch (tuple[torch.Tensor, ...]): Batch of data from the DataLoader.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Loss and metrics for the batch.
        """
        with torch.no_grad():
            loss_metrics = self.loss_fn(self.model, batch)
        return self._modify_batch_end_loss_metrics(loss_metrics)

    def _batch_iter(
        self,
        dataloader: DataLoader | DictDataLoader,
        num_batches: int,
    ) -> Iterable[tuple[torch.Tensor, ...] | None]:
        """
        Yields batches from the provided dataloader.

        Args:
            dataloader ([DataLoader]): The dataloader to iterate over.
            num_batches (int): The maximum number of batches to yield.

        Yields:
            Iterable[tuple[torch.Tensor, ...] | None]: A batch from the dataloader moved to the
                specified device and dtype.
        """
        if dataloader is None:
            for _ in range(num_batches):
                yield None
        else:
            for batch in islice(dataloader, num_batches):
                # batch is moved to device inside optimize step
                # batch = data_to_device(batch, device=self.device, dtype=self.data_dtype)
                yield batch

    def _modify_batch_end_loss_metrics(
        self, loss_metrics: tuple[torch.Tensor, dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Modifies the loss and metrics at the end of batch for proper logging.

        All metrics are prefixed with the proper state of the training process
         - "train_" or "val_" or "test_"
        A "{state}_loss" is added to metrics.

        Args:
            loss_metrics (tuple[torch.Tensor, dict[str, Any]]): Original loss and metrics.

        Returns:
            tuple[None | torch.Tensor, dict[str, Any]]: Modified loss and metrics.
        """
        for phase in ["train", "val", "test"]:
            if phase in self.training_stage:
                loss, metrics = loss_metrics
                updated_metrics = {f"{phase}_{key}": value for key, value in metrics.items()}
                updated_metrics[f"{phase}_loss"] = loss
                return loss, updated_metrics
        return loss_metrics

    def _reset_model_and_opt(self) -> None:
        """
        Save model_old and optimizer_old for epoch 0.

        This allows us to create a copy of model
        and optimizer before running the optimization.

        We do this because optimize step provides loss, metrics
        before step of optimization
        To align them with model/optimizer correctly, we checkpoint
        the older copy of the model.
        """

        # TODO: review optimize_step to provide iteration aligned model and loss.
        try:
            # Deep copy model and optimizer to maintain checkpoints
            self.model_old = copy.deepcopy(self.model)
            self.optimizer_old = copy.deepcopy(self.optimizer)
        except Exception:
            self.model_old = self.model
            self.optimizer_old = self.optimizer

    def build_optimize_result(
        self,
        result: None
        | tuple[torch.Tensor, dict[Any, Any]]
        | list[tuple[torch.Tensor, dict[Any, Any]]]
        | list[list[tuple[torch.Tensor, dict[Any, Any]]]],
    ) -> None:
        """
        Builds and stores the optimization result by calculating the average loss and metrics.

        Result (or loss_metrics) can have multiple formats:
        - `None` Indicates no loss or metrics data is provided.
        - `tuple[torch.Tensor, dict[str, Any]]` A single tuple containing the loss tensor
            and metrics dictionary - at the end of batch.
        - `list[tuple[torch.Tensor, dict[str, Any]]]` A list of tuples for
            multiple batches.
        - `list[list[tuple[torch.Tensor, dict[str, Any]]]]` A list of lists of tuples,
        where each inner list represents metrics across multiple batches within an epoch.

        Args:
            result: (None |
                    tuple[torch.Tensor, dict[Any, Any]] |
                    list[tuple[torch.Tensor, dict[Any, Any]]] |
                    list[list[tuple[torch.Tensor, dict[Any, Any]]]])
                        The loss and metrics data, which can have multiple formats

        Returns:
            None: This method does not return anything. It sets `self.opt_result` with
            the computed average loss and metrics.
        """
        loss_metrics = result
        if loss_metrics is None:
            loss = None
            metrics: dict[Any, Any] = {}
        elif isinstance(loss_metrics, tuple):
            # Single tuple case
            loss, metrics = loss_metrics
        else:
            last_epoch: list[tuple[torch.Tensor, dict[Any, Any]]] = []
            if isinstance(loss_metrics, list):
                # Check if it's a list of tuples
                if all(isinstance(item, tuple) for item in loss_metrics):
                    last_epoch = cast(list[tuple[torch.Tensor, dict[Any, Any]]], loss_metrics)
                # Check if it's a list of lists of tuples
                elif all(isinstance(item, list) for item in loss_metrics):
                    last_epoch = cast(
                        list[tuple[torch.Tensor, dict[Any, Any]]],
                        loss_metrics[-1] if loss_metrics else [],
                    )
                else:
                    raise ValueError(
                        "Invalid format for result: Expected None, tuple, list of tuples,"
                        " or list of lists of tuples."
                    )

            if not last_epoch:
                loss, metrics = None, {}
            else:
                # Compute the average loss over the batches
                loss_tensor = torch.stack([loss_batch for loss_batch, _ in last_epoch])
                avg_loss = loss_tensor.mean()

                # Collect and average metrics for all batches
                metric_keys = last_epoch[0][1].keys()
                metrics_stacked: dict = {key: [] for key in metric_keys}

                for _, metrics_batch in last_epoch:
                    for key in metric_keys:
                        value = metrics_batch[key]
                        metrics_stacked[key].append(value)

                avg_metrics = {key: torch.stack(metrics_stacked[key]).mean() for key in metric_keys}

                loss, metrics = avg_loss, avg_metrics

        # Store the optimization result
        self.opt_result = OptimizeResult(
            self.current_epoch, self.model_old, self.optimizer_old, loss, metrics
        )
