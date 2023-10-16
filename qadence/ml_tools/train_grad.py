from __future__ import annotations

from typing import Callable

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.logger import get_logger
from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.printing import print_metrics, write_tensorboard
from qadence.ml_tools.saveload import load_checkpoint, write_checkpoint

logger = get_logger(__name__)


def train(
    model: Module,
    dataloader: DictDataLoader | DataLoader | list[Tensor] | tuple[Tensor, Tensor] | None,
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: str = "cpu",
    optimize_step: Callable = optimize_step,
    write_tensorboard: Callable = write_tensorboard,
) -> tuple[Module, Optimizer]:
    """Runs the training loop with gradient-based optimizer

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
            optimizer, loss_fn, xs, device="cpu")` (see the example below).
            Apart from the default we already supply three other optimization
            functions `optimize_step_evo`, `optimize_step_grad_norm`, and
            `optimize_step_inv_dirichlet`. Learn more about how to use this in
            the [Advancded features](../../tutorials/advanced) tutorial of the
            documentation.
        write_tensorboard: Customizable tensorboard logging callback which is
            called every `config.write_every` iterations. The function must have
            the signature `write_tensorboard(writer, loss, metrics, iteration)`
            (see the example below).

    Example:
    ```python exec="on" source="material-block"
    from pathlib import Path
    import torch
    from itertools import count
    from qadence.constructors import hamiltonian_factory, hea, feature_map
    from qadence import chain, Parameter, QuantumCircuit, Z
    from qadence.models import QNN
    from qadence.ml_tools import train_with_grad, TrainConfig

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
    config = TrainConfig(
        folder=tmp_path,
        max_iter=n_epochs,
        checkpoint_every=100,
        write_every=100,
        batch_size=batch_size,
    )
    batch_size = 25
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    train_with_grad(model, (x, y), optimizer, config, loss_fn=loss_fn)
    ```
    """

    assert loss_fn is not None, "Provide a valid loss function"

    # Move model to device before optimizer is loaded
    model = model.to(device)

    # load available checkpoint
    init_iter = 0
    if config.folder:
        model, optimizer, init_iter = load_checkpoint(config.folder, model, optimizer)
        logger.debug(f"Loaded model and optimizer from {config.folder}")
    # initialize tensorboard
    writer = SummaryWriter(config.folder, purge_step=init_iter)

    ## Training
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    if isinstance(dataloader, (list, tuple)):
        from qadence.ml_tools.data import to_dataloader

        assert len(dataloader) == 2, "Please provide exactly two torch tensors."
        x, y = dataloader
        dataloader = to_dataloader(x=x, y=y, batch_size=config.batch_size)
    with progress:
        dl_iter = iter(dataloader) if isinstance(dataloader, DictDataLoader) else None

        # outer epoch loop
        for iteration in progress.track(range(init_iter, init_iter + config.max_iter)):
            try:
                # in case there is not data needed by the model
                # this is the case, for example, of quantum models
                # which do not have classical input data (e.g. chemistry)
                if dataloader is None:
                    loss, metrics = optimize_step(
                        model, optimizer, loss_fn, dataloader, device=device
                    )
                    loss = loss.item()

                # single epoch with DictDataloader using a single iteration method
                # DictDataloader returns a single sample of the data
                # with a given batch size decided when the dataloader is defined
                elif isinstance(dataloader, DictDataLoader):
                    # resample all the time from the dataloader
                    # by creating a fresh iterator if the dataloader
                    # does not support automatically iterating datasets
                    if not dataloader.has_automatic_iter:
                        dl_iter = iter(dataloader)
                    data = next(dl_iter)  # type: ignore[arg-type]
                    loss, metrics = optimize_step(model, optimizer, loss_fn, data, device=device)

                elif isinstance(dataloader, DataLoader):
                    # single-epoch with standard DataLoader
                    # otherwise a standard PyTorch DataLoader behavior
                    # is assumed with optional mini-batches
                    running_loss = 0.0
                    for i, data in enumerate(dataloader):
                        # TODO: make sure to average metrics as well
                        loss, metrics = optimize_step(
                            model, optimizer, loss_fn, data, device=device
                        )
                        running_loss += loss.item()
                    loss = running_loss / (i + 1)

                else:
                    raise NotImplementedError("Unsupported dataloader type!")

                if iteration % config.print_every == 0:
                    print_metrics(loss, metrics, iteration)

                if iteration % config.write_every == 0:
                    write_tensorboard(writer, loss, metrics, iteration)

                if config.folder:
                    if iteration % config.checkpoint_every == 0:
                        write_checkpoint(config.folder, model, optimizer, iteration)

            except KeyboardInterrupt:
                print("Terminating training gracefully after the current iteration.")
                break

    # Final writing and checkpointing
    if config.folder:
        write_checkpoint(config.folder, model, optimizer, iteration)
    write_tensorboard(writer, loss, metrics, iteration)
    writer.close()

    return model, optimizer
