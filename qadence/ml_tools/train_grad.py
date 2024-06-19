from __future__ import annotations

import math
from logging import getLogger
from typing import Callable, Union

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import complex128, float32, float64
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import DataParallel, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, data_to_device
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.printing import print_metrics, write_tensorboard
from qadence.ml_tools.saveload import load_checkpoint, write_checkpoint

logger = getLogger(__name__)


def train(
    model: Module,
    dataloader: Union[None, DataLoader, DictDataLoader],
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: torch_device = None,
    optimize_step: Callable = optimize_step,
    write_tensorboard: Callable = write_tensorboard,
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
        loss_fn: Loss function returning (loss: float, metrics: dict[str, float])
        device: String defining device to train on, pass 'cuda' for GPU.
        optimize_step: Customizable optimization callback which is called at every iteration.=
            The function must have the signature `optimize_step(model,
            optimizer, loss_fn, xs, device="cpu")`.
        write_tensorboard: Customizable tensorboard logging callback which is
            called every `config.write_every` iterations. The function must have
            the signature `write_tensorboard(writer, loss, metrics, iteration)`
            (see the example below).
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
    if config.folder:
        model, optimizer, init_iter = load_checkpoint(config.folder, model, optimizer)
        logger.debug(f"Loaded model and optimizer from {config.folder}")

    # Move model to device before optimizer is loaded
    if isinstance(model, DataParallel):
        model = model.module.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device, dtype=dtype)
    # initialize tensorboard
    writer = SummaryWriter(config.folder, purge_step=init_iter)

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
    with progress:
        dl_iter = iter(dataloader) if dataloader is not None else None
        if perform_val:
            dl_iter_val = iter(val_dataloader) if val_dataloader is not None else None

        # outer epoch loop
        for iteration in progress.track(range(init_iter, init_iter + config.max_iter)):
            try:
                # in case there is not data needed by the model
                # this is the case, for example, of quantum models
                # which do not have classical input data (e.g. chemistry)
                if dataloader is None:
                    loss, metrics = optimize_step(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        xs=None,
                        device=device,
                        dtype=data_dtype,
                    )
                    loss = loss.item()

                elif isinstance(dataloader, (DictDataLoader, DataLoader)):
                    loss, metrics = optimize_step(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        xs=next(dl_iter),  # type: ignore[arg-type]
                        device=device,
                        dtype=data_dtype,
                    )

                else:
                    raise NotImplementedError(
                        f"Unsupported dataloader type: {type(dataloader)}. "
                        "You can use e.g. `qadence.ml_tools.to_dataloader` to build a dataloader."
                    )

                if iteration % config.print_every == 0 and config.verbose:
                    print_metrics(loss, metrics, iteration)

                if iteration % config.write_every == 0:
                    write_tensorboard(writer, loss, metrics, iteration)

                if perform_val:
                    if iteration % config.val_every == 0:
                        xs = next(dl_iter_val)
                        xs_to_device = data_to_device(xs, device=device, dtype=data_dtype)
                        val_loss, _ = loss_fn(model, xs_to_device)
                        if config.validation_criterion(val_loss, best_val_loss, config.val_epsilon):  # type: ignore[misc]
                            best_val_loss = val_loss
                            if config.folder and config.checkpoint_best_only:
                                write_checkpoint(config.folder, model, optimizer, iteration="best")
                            metrics["val_loss"] = val_loss
                            write_tensorboard(writer, math.nan, metrics, iteration)

                if config.folder:
                    if iteration % config.checkpoint_every == 0 and not config.checkpoint_best_only:
                        write_checkpoint(config.folder, model, optimizer, iteration)

            except KeyboardInterrupt:
                logger.info("Terminating training gracefully after the current iteration.")
                break

    # Final writing and checkpointing
    if config.folder and not config.checkpoint_best_only:
        write_checkpoint(config.folder, model, optimizer, iteration)
    write_tensorboard(writer, loss, metrics, iteration)
    writer.close()

    return model, optimizer
