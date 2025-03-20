from __future__ import annotations

import copy
from itertools import islice
from logging import getLogger
from typing import Any, Callable, Iterable, cast
from nevergrad.optimization.base import Optimizer as NGOptimizer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, OptimizeResult, data_to_device
from qadence.ml_tools.information import InformationContent
from qadence.ml_tools.optimize_step import optimize_step, update_ng_parameters
from qadence.ml_tools.stages import TrainingStage

from .train_utils.base_trainer import BaseTrainer
from .train_utils.accelerator import Accelerator

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
        self._stop_training: torch.Tensor = torch.tensor(0, dtype=torch.int)
        self.progress: Progress | None = None

        # Integration with Accelerator:
        self.accelerator = Accelerator(
            backend=config.backend,
            nprocs=config.nprocs,
            compute_setup=config.compute_setup,
            dtype=config.dtype,
            log_setup=config.log_setup,
        )
        # Decorate the unbound Trainer.fit method with accelerator.distribute.
        # We use __get__ to bind the decorated method to the current instance,
        # ensuring that 'self' is passed only once when self.fit is called.
        self.fit = self.accelerator.distribute(Trainer.fit).__get__(self, Trainer)  # type: ignore[method-assign]

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
        self._stop_training = torch.tensor(
            0, dtype=torch.int, device=self.accelerator.execution.device
        )
        # initalize config in the first process, and broadcast it to all processes
        if self.accelerator.rank == 0:
            self.config_manager.initialize_config()
        self.config_manager = self.accelerator.broadcast(self.config_manager, src=0)
        self.callback_manager.start_training(trainer=self)

        # Integration with Accelerator: prepare the model, optimizer, and dataloaders.
        (self.model, self.optimizer, self.train_dataloader, self.val_dataloader) = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader
            )
        )

        # Progress bar for training visualization
        if self.accelerator.world_size == 1:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            )

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
        Runs the main training loop over multiple epochs.

        This method sets up the training process by performing any necessary pre-training
        actions (via `on_train_start`), configuring progress tracking (if available), and then
        iteratively calling `_train_epoch` to run through the epochs.

        Returns:
            list[list[tuple[torch.Tensor, dict[str, Any]]]]: Training loss
            metrics for all epochs.
                list    -> list                  -> tuples
                Epochs  -> Training Batches      -> (loss, metrics)
        """
        self.on_train_start()
        epoch_start, epoch_end = (
            self.global_step,
            self.global_step + self.config_manager.config.max_iter + 1,
        )

        if self.accelerator.world_size == 1 and self.progress:
            # Progress setup is only available for non-spawned training.
            with self.progress:
                train_task = self.progress.add_task(
                    "Training", total=self.config_manager.config.max_iter
                )
                if self.perform_val:
                    val_task = self.progress.add_task(
                        "Validation",
                        total=(self.config_manager.config.max_iter + 1) / self.config.val_every,
                    )
                else:
                    val_task = None
                train_losses, val_losses = self._train_epochs(
                    epoch_start, epoch_end, train_task, val_task
                )
        else:
            train_losses, val_losses = self._train_epochs(epoch_start, epoch_end)

        self.on_train_end(train_losses, val_losses)
        return train_losses

    def _train_epochs(
        self,
        epoch_start: int,
        epoch_end: int,
        train_task: int | None = None,
        val_task: int | None = None,
    ) -> tuple[
        list[list[tuple[torch.Tensor, dict[str, Any]]]],
        list[list[tuple[torch.Tensor, dict[str, Any]]]],
    ]:
        """
        Executes the training loop for a series of epochs.

        Args:
            epoch_start (int): The starting epoch index.
            epoch_end (int): The ending epoch index (non-inclusive).
            train_task (int | None, optional): The progress bar task ID for training updates.
                If provided, the progress bar will be updated after each epoch. Defaults to None.
            val_task (int | None, optional): The progress bar task ID for validation updates.
                If provided and validation is enabled, the progress bar will be updated after each validation run.
                Defaults to None.

        Returns:
            list[list[tuple[torch.Tensor, dict[str, Any]]]]: A tuple of
            Training loss metrics for all epochs.
                list    -> list                  -> tuples
                Epochs  -> Training Batches      -> (loss, metrics)
            And Validation loss metrics for all epochs
                list    -> list                  -> tuples
                Epochs  -> Training Batches      -> (loss, metrics)
        """
        train_losses = []
        val_losses = []

        # Iterate over the epochs
        for epoch in range(epoch_start, epoch_end):
            if not self.stop_training():
                try:
                    self.current_epoch = epoch
                    self.on_train_epoch_start()
                    train_epoch_loss_metrics = self.run_training(self.train_dataloader)
                    train_losses.append(train_epoch_loss_metrics)
                    self.on_train_epoch_end(train_epoch_loss_metrics)

                    # Run validation periodically if specified
                    if self.perform_val and (epoch % self.config.val_every == 0):
                        self.on_val_epoch_start()
                        val_epoch_loss_metrics = self.run_validation(self.val_dataloader)
                        val_losses.append(val_epoch_loss_metrics)
                        self.on_val_epoch_end(val_epoch_loss_metrics)
                        if val_task is not None:
                            self.progress.update(val_task, advance=1)  # type: ignore[union-attr]

                    if train_task is not None:
                        self.progress.update(train_task, advance=1)  # type: ignore[union-attr]
                except KeyboardInterrupt:
                    self._stop_training.fill_(1)
            else:
                if self.accelerator.rank == 0:
                    logger.info("Terminating training gracefully after the current iteration.")
                self.accelerator.finalize()
                break
        return train_losses, val_losses

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

        for batch in self._batch_iter(dataloader, self.num_training_batches):
            self.on_train_batch_start(batch)
            train_batch_loss_metrics = self.run_train_batch(batch)
            if self.config.all_reduce_metrics:
                train_batch_loss_metrics = self._aggregate_result(train_batch_loss_metrics)
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
                device=self.accelerator.execution.device,
                dtype=self.accelerator.execution.data_dtype,
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
            if self.config.all_reduce_metrics:
                val_batch_loss_metrics = self._aggregate_result(val_batch_loss_metrics)
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

        The batch of data is also moved
        to the correct device and dtype using accelerator.prepare.

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
                yield self.accelerator.prepare_batch(batch)

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

    def _aggregate_result(
        self, result: tuple[torch.Tensor, dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Aggregates the loss and metrics using the Accelerator's all_reduce_dict method if aggregation is enabled.

        Args:
            result:     (tuple[torch.Tensor, dict[str, Any]])
                            The result consisting of loss and metrics.For more details,
                            look at the signature of build_optimize_result.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: The aggregated loss and metrics.
        """
        loss, metrics = result
        if self.config.all_reduce_metrics:
            reduced = self.accelerator.all_reduce_dict({"loss": loss, **metrics})
            loss = reduced.pop("loss")
            metrics = reduced
            return loss, metrics
        else:
            return loss, metrics

    def stop_training(self) -> bool:
        """
        Helper function to indicate if the training should be stopped.

        We all_reduce the indicator across all processes to ensure all processes are stopped.

        Notes:
            self._stop_training indicator indicates if the training should be stopped.
            0 is continue. 1 is stop.
        """
        _stop_training = self.accelerator.all_reduce_dict(
            {"indicator": self._stop_training}, op="max"
        )
        return bool(_stop_training["indicator"] > 0)

    def build_optimize_result(
        self,
        result: (
            None
            | tuple[torch.Tensor, dict[Any, Any]]
            | list[tuple[torch.Tensor, dict[Any, Any]]]
            | list[list[tuple[torch.Tensor, dict[Any, Any]]]]
        ),
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
            self.current_epoch,
            self.model,
            self.optimizer,
            loss,
            metrics,
            rank=self.accelerator.rank,
            device=self.accelerator.execution.device,
        )

    def get_ic_grad_bounds(
        self,
        eta: float,
        epsilons: torch.Tensor,
        variation_multiple: int = 20,
        dataloader: DataLoader | DictDataLoader | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the bounds on the gradient norm of the loss using Information Content.

        Args:
            eta (float): The sensitivity IC.
            epsilons (torch.Tensor): The epsilons to use for thresholds to for discretization of the
                finite derivatives.
            variation_multiple (int): The number of sets of variational parameters to generate per
                each variational parameter. The number of variational parameters required for the
                statisctiacal analysis scales linearly with the amount of them present in the
                model. This is that linear factor.
            dataloader (DataLoader | DictDataLoader | None): The dataloader for training data. A
                new dataloader can be provided, or the dataloader provided in the trinaer will be
                used. In case no dataloaders are provided at either places, it assumes that the
                model does not require any input data.

        Returns:
            tuple[float, float, float]: The max IC lower bound, max IC upper bound, and sensitivity
                IC upper bound.

        Examples:
            ```python
            import torch
            from torch.optim.adam import Adam

            from qadence.constructors import ObservableConfig
            from qadence.ml_tools.config import AnsatzConfig, FeatureMapConfig, TrainConfig
            from qadence.ml_tools.data import to_dataloader
            from qadence.ml_tools.models import QNN
            from qadence.ml_tools.optimize_step import optimize_step
            from qadence.ml_tools.trainer import Trainer
            from qadence.operations.primitive import Z

            fm_config = FeatureMapConfig(num_features=1)
            ansatz_config = AnsatzConfig(depth=4)
            obs_config = ObservableConfig(detuning=Z)

            qnn = QNN.from_configs(
                register=4,
                obs_config=obs_config,
                fm_config=fm_config,
                ansatz_config=ansatz_config,
            )

            optimizer = Adam(qnn.parameters(), lr=0.001)

            batch_size = 25
            x = torch.linspace(0, 1, 32).reshape(-1, 1)
            y = torch.sin(x)
            train_loader = to_dataloader(x, y, batch_size=batch_size, infinite=True)

            train_config = TrainConfig(max_iter=100)

            trainer = Trainer(
                model=qnn,
                optimizer=optimizer,
                config=train_config,
                loss_fn="mse",
                train_dataloader=train_loader,
                optimize_step=optimize_step,
            )

            # Perform exploratory landscape analysis with Information Content
            ic_sensitivity_threshold = 1e-4
            epsilons = torch.logspace(-2, 2, 10)

            max_ic_lower_bound, max_ic_upper_bound, sensitivity_ic_upper_bound = (
                trainer.get_ic_grad_bounds(
                    eta=ic_sensitivity_threshold,
                    epsilons=epsilons,
                )
            )

            # Resume training as usual...

            trainer.fit(train_loader)
            ```
        """
        if not self._use_grad:
            logger.warning(
                "Gradient norm bounds are only relevant when using a gradient based optimizer. \
                    Currently the trainer is set to use a gradient-free optimizer."
            )

        dataloader = dataloader if dataloader is not None else self.train_dataloader

        batch = next(iter(self._batch_iter(dataloader, num_batches=1)))

        ic = InformationContent(self.model, self.loss_fn, batch, epsilons)

        max_ic_lower_bound, max_ic_upper_bound = ic.get_grad_norm_bounds_max_IC()
        sensitivity_ic_upper_bound = ic.get_grad_norm_bounds_sensitivity_IC(eta)

        return max_ic_lower_bound, max_ic_upper_bound, sensitivity_ic_upper_bound
