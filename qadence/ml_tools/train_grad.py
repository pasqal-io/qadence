from __future__ import annotations

import importlib
import math
from logging import getLogger
from typing import Any, Callable, Union

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor, complex128, float32, float64
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import DataParallel, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import Callback, TrainConfig, run_callbacks
from qadence.ml_tools.data import DictDataLoader, OptimizeResult, data_to_device
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.printing import (
    log_model_tracker,
    log_tracker,
    plot_tracker,
    print_metrics,
    write_tracker,
)
from qadence.ml_tools.saveload import load_checkpoint, write_checkpoint
from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)


def train(
    model: Module,
    dataloader: Union[None, DataLoader, DictDataLoader],
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: torch_device = None,
    optimize_step: Callable = optimize_step,
    dtype: torch_dtype = None,
) -> tuple[Module, Optimizer]:
    """Runs the training loop with gradient-based optimizer.

    Assumes that `loss_fn` returns a tuple of (loss,
    metrics: dict), where `metrics` is a dict of scalars. Loss and metrics are
    written to tensorboard. Checkpoints are written every
    `config.checkpoint_every` steps (and after the last training step).  If a
    checkpoint is found at `config.folder` we resume training from there.  The
    tensorboard logs can be viewed via `tensorboard --logdir /path/to/folder`.

    Args:
        model: The model to train.
        dataloader: dataloader of different types. If None, no data is required by
            the model
        optimizer: The optimizer to use.
        config: `TrainConfig` with additional training options.
        loss_fn: Loss function returning (loss: float, metrics: dict[str, float], ...)
        device: String defining device to train on, pass 'cuda' for GPU.
        optimize_step: Customizable optimization callback which is called at every iteration.=
            The function must have the signature `optimize_step(model,
            optimizer, loss_fn, xs, device="cpu")`.
        dtype: The dtype to use for the data.

    Example:
    ```python exec="on" source="material-block"
    from pathlib import Path
    import torch
    from itertools import count
    from qadence import Parameter, QuantumCircuit, Z
    from qadence import hamiltonian_factory, hea, feature_map, chain
    from qadence import QNN
    from qadence.ml_tools import TrainConfig, train_with_grad, to_dataloader

    n_qubits = 2
    fm = feature_map(n_qubits)
    ansatz = hea(n_qubits=n_qubits, depth=3)
    observable = hamiltonian_factory(n_qubits, detuning = Z)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)

    model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)

    ## lets prepare the train routine

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x, y = data[0], data[1]
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    tmp_path = Path("/tmp")
    n_epochs = 5
    batch_size = 25
    config = TrainConfig(
        folder=tmp_path,
        max_iter=n_epochs,
        checkpoint_every=100,
        write_every=100,
    )
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    data = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    ```
    """
    # load available checkpoint
    init_iter = 0
    log_device = "cpu" if device is None else device
    if config.folder:
        model, optimizer, init_iter = load_checkpoint(
            config.folder, model, optimizer, device=log_device
        )
        logger.debug(f"Loaded model and optimizer from {config.folder}")

    # Move model to device before optimizer is loaded
    if isinstance(model, DataParallel):
        model = model.module.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    # initialize tracking tool
    if config.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
        writer = SummaryWriter(config.folder, purge_step=init_iter)
    else:
        writer = importlib.import_module("mlflow")

    perform_val = isinstance(config.val_every, int)
    if perform_val:
        if not isinstance(dataloader, DictDataLoader):
            raise ValueError(
                "If `config.val_every` is provided as an integer, dataloader must"
                "be an instance of `DictDataLoader`."
            )
        iter_keys = dataloader.dataloaders.keys()
        if "train" not in iter_keys or "val" not in iter_keys:
            raise ValueError(
                "If `config.val_every` is provided as an integer, the dictdataloader"
                "must have `train` and `val` keys to access the respective dataloaders."
            )
        val_dataloader = dataloader.dataloaders["val"]
        dataloader = dataloader.dataloaders["train"]

    ## Training
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    data_dtype = None
    if dtype:
        data_dtype = float64 if dtype == complex128 else float32

    best_val_loss = math.inf

    if not ((dataloader is None) or isinstance(dataloader, (DictDataLoader, DataLoader))):
        raise NotImplementedError(
            f"Unsupported dataloader type: {type(dataloader)}. "
            "You can use e.g. `qadence.ml_tools.to_dataloader` to build a dataloader."
        )

    def next_loss_iter(dl_iter: Union[None, DataLoader, DictDataLoader]) -> Any:
        """Get loss on the next batch of a dataloader.

            loaded on device if not None.

        Args:
            dl_iter (Union[None, DataLoader, DictDataLoader]): Dataloader.

        Returns:
            Any: Loss value
        """
        xs = next(dl_iter) if dl_iter is not None else None
        xs_to_device = data_to_device(xs, device=device, dtype=data_dtype)
        return loss_fn(model, xs_to_device)

    # populate callbacks with already available internal functions
    # printing, writing and plotting
    callbacks = config.callbacks

    # printing
    if config.verbose and config.print_every > 0:
        # Note that the loss returned by optimize_step
        # is the value before doing the training step
        # which is printed accordingly by the previous iteration number
        callbacks += [
            Callback(
                lambda opt_res: print_metrics(opt_res.loss, opt_res.metrics, opt_res.iteration - 1),
                called_every=config.print_every,
            )
        ]

    # plotting
    callbacks += [
        Callback(
            lambda opt_res: plot_tracker(
                writer,
                opt_res.model,
                opt_res.iteration,
                config.plotting_functions,
                tracking_tool=config.tracking_tool,
            ),
            called_every=config.plot_every,
            call_before_opt=True,
        )
    ]

    # writing metrics
    # we specify two writers,
    # to write at evaluation time and before evaluation
    callbacks += [
        Callback(
            lambda opt_res: write_tracker(
                writer,
                opt_res.loss,
                opt_res.metrics,
                opt_res.iteration - 1,  # loss returned be optimized_step is at -1
                tracking_tool=config.tracking_tool,
            ),
            called_every=config.write_every,
            call_end_epoch=True,
        ),
        Callback(
            lambda opt_res: write_tracker(
                writer,
                opt_res.loss,
                opt_res.metrics,
                opt_res.iteration,  # after_opt we match the right loss function
                tracking_tool=config.tracking_tool,
            ),
            called_every=config.write_every,
            call_end_epoch=False,
            call_after_opt=True,
        ),
    ]
    if perform_val:
        callbacks += [
            Callback(
                lambda opt_res: write_tracker(
                    writer,
                    None,
                    opt_res.metrics,
                    opt_res.iteration,
                    tracking_tool=config.tracking_tool,
                ),
                called_every=config.write_every,
                call_before_opt=True,
                call_during_eval=True,
            )
        ]

    # checkpointing
    if config.folder and config.checkpoint_every > 0 and not config.checkpoint_best_only:
        callbacks += [
            Callback(
                lambda opt_res: write_checkpoint(
                    config.folder,  # type: ignore[arg-type]
                    opt_res.model,
                    opt_res.optimizer,
                    opt_res.iteration,
                ),
                called_every=config.checkpoint_every,
                call_before_opt=False,
                call_after_opt=True,
            )
        ]

    if config.folder and config.checkpoint_best_only:
        callbacks += [
            Callback(
                lambda opt_res: write_checkpoint(
                    config.folder,  # type: ignore[arg-type]
                    opt_res.model,
                    opt_res.optimizer,
                    "best",
                ),
                called_every=config.checkpoint_every,
                call_before_opt=True,
                call_after_opt=True,
                call_during_eval=True,
            )
        ]

    callbacks_before_opt = [
        callback
        for callback in callbacks
        if callback.call_before_opt and not callback.call_during_eval
    ]
    callbacks_before_opt_eval = [
        callback for callback in callbacks if callback.call_before_opt and callback.call_during_eval
    ]

    with progress:
        dl_iter = iter(dataloader) if dataloader is not None else None

        # Initial validation evaluation
        try:
            opt_result = OptimizeResult(init_iter, model, optimizer)
            if perform_val:
                dl_iter_val = iter(val_dataloader) if val_dataloader is not None else None
                best_val_loss, metrics, *_ = next_loss_iter(dl_iter_val)
                metrics["val_loss"] = best_val_loss
                opt_result.metrics = metrics
                run_callbacks(callbacks_before_opt_eval, opt_result)

            run_callbacks(callbacks_before_opt, opt_result)

        except KeyboardInterrupt:
            logger.info("Terminating training gracefully after the current iteration.")

        # outer epoch loop
        init_iter += 1
        callbacks_end_epoch = [
            callback
            for callback in callbacks
            if callback.call_end_epoch and not callback.call_during_eval
        ]
        callbacks_end_epoch_eval = [
            callback
            for callback in callbacks
            if callback.call_end_epoch and callback.call_during_eval
        ]
        for iteration in progress.track(range(init_iter, init_iter + config.max_iter)):
            try:
                # in case there is not data needed by the model
                # this is the case, for example, of quantum models
                # which do not have classical input data (e.g. chemistry)
                loss, metrics = optimize_step(
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    xs=None if dataloader is None else next(dl_iter),  # type: ignore[arg-type]
                    device=device,
                    dtype=data_dtype,
                )
                if isinstance(loss, Tensor):
                    loss = loss.item()
                opt_result = OptimizeResult(iteration, model, optimizer, loss, metrics)
                run_callbacks(callbacks_end_epoch, opt_result)

                if perform_val:
                    if iteration % config.val_every == 0:
                        val_loss, *_ = next_loss_iter(dl_iter_val)
                        if config.validation_criterion(val_loss, best_val_loss, config.val_epsilon):  # type: ignore[misc]
                            best_val_loss = val_loss
                            metrics["val_loss"] = val_loss
                            opt_result.metrics = metrics

                            run_callbacks(callbacks_end_epoch_eval, opt_result)

            except KeyboardInterrupt:
                logger.info("Terminating training gracefully after the current iteration.")
                break

        # For handling printing/writing the last training loss
        # as optimize_step does not give the loss value at the last iteration
        try:
            loss, metrics, *_ = next_loss_iter(dl_iter)
            if isinstance(loss, Tensor):
                loss = loss.item()
            if perform_val:
                # reputting val_loss as already evaluated before
                metrics["val_loss"] = val_loss
            print_metrics(loss, metrics, iteration)

        except KeyboardInterrupt:
            logger.info("Terminating training gracefully after the current iteration.")

    # Final callbacks, by default checkpointing and writing
    opt_result = OptimizeResult(iteration, model, optimizer, loss, metrics)
    callbacks_after_opt = [callback for callback in callbacks if callback.call_after_opt]
    run_callbacks(callbacks_after_opt, opt_result, is_last_iteration=True)

    # writing hyperparameters
    if config.hyperparams:
        log_tracker(writer, config.hyperparams, metrics, tracking_tool=config.tracking_tool)

    # logging the model
    if config.log_model:
        log_model_tracker(writer, model, dataloader, tracking_tool=config.tracking_tool)

    # close tracker
    if config.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
        writer.close()
    elif config.tracking_tool == ExperimentTrackingTool.MLFLOW:
        writer.end_run()

    return model, optimizer
