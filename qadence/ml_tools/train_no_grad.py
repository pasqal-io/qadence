from __future__ import annotations

import importlib
from logging import getLogger
from typing import Callable

import nevergrad as ng
from nevergrad.optimization.base import Optimizer as NGOptimizer
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader
from qadence.ml_tools.parameters import get_parameters, set_parameters
from qadence.ml_tools.printing import (
    log_model_tracker,
    log_tracker,
    plot_tracker,
    print_metrics,
    write_tracker,
)
from qadence.ml_tools.saveload import load_checkpoint, write_checkpoint
from qadence.ml_tools.tensors import promote_to_tensor
from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)


def train(
    model: Module,
    dataloader: DictDataLoader | DataLoader | None,
    optimizer: NGOptimizer,
    config: TrainConfig,
    loss_fn: Callable,
) -> tuple[Module, NGOptimizer]:
    """Runs the training loop with a gradient-free optimizer.

    Assumes that `loss_fn` returns a tuple of (loss, metrics: dict), where
    `metrics` is a dict of scalars. Loss and metrics are written to
    tensorboard. Checkpoints are written every `config.checkpoint_every` steps
    (and after the last training step).  If a checkpoint is found at `config.folder`
    we resume training from there.  The tensorboard logs can be viewed via
    `tensorboard --logdir /path/to/folder`.

    Args:
        model: The model to train
        dataloader: Dataloader constructed via `dictdataloader`
        optimizer: The optimizer to use taken from the Nevergrad library. If this is not
            the case the function will raise an AssertionError
        loss_fn: Loss function returning (loss: float, metrics: dict[str, float])
    """
    init_iter = 0
    if config.folder:
        model, optimizer, init_iter = load_checkpoint(config.folder, model, optimizer)
        logger.debug(f"Loaded model and optimizer from {config.folder}")

    def _update_parameters(
        data: Tensor | None, ng_params: ng.p.Array
    ) -> tuple[float, dict, ng.p.Array]:
        loss, metrics = loss_fn(model, data)  # type: ignore[misc]
        optimizer.tell(ng_params, float(loss))
        ng_params = optimizer.ask()  # type: ignore [assignment]
        params = promote_to_tensor(ng_params.value, requires_grad=False)
        set_parameters(model, params)
        return loss, metrics, ng_params

    assert loss_fn is not None, "Provide a valid loss function"
    # TODO: support also Scipy optimizers
    assert isinstance(optimizer, NGOptimizer), "Use only optimizers from the Nevergrad library"

    # initialize tracking tool
    if config.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
        writer = SummaryWriter(config.folder, purge_step=init_iter)
    else:
        writer = importlib.import_module("mlflow")

        # writer.mlflow.pytorch.autolog(
        #     log_every_n_step=config.write_every, log_models=False, log_datasets=False
        # )

    # set optimizer configuration and initial parameters
    optimizer.budget = config.max_iter
    optimizer.enable_pickling()

    # TODO: Make it GPU compatible if possible
    params = get_parameters(model).detach().numpy()
    ng_params = ng.p.Array(init=params)

    # serial training
    # TODO: Add a parallelization using the num_workers argument in Nevergrad
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    with progress:
        dl_iter = iter(dataloader) if dataloader is not None else None

        for iteration in progress.track(range(init_iter, init_iter + config.max_iter)):
            if dataloader is None:
                loss, metrics, ng_params = _update_parameters(None, ng_params)

            elif isinstance(dataloader, (DictDataLoader, DataLoader)):
                data = next(dl_iter)  # type: ignore[arg-type]
                loss, metrics, ng_params = _update_parameters(data, ng_params)

            else:
                raise NotImplementedError("Unsupported dataloader type!")

            if iteration % config.print_every == 0 and config.verbose:
                print_metrics(loss, metrics, iteration)

            if iteration % config.write_every == 0:
                write_tracker((writer, loss, metrics, iteration), config.tracking_tool)

            if iteration % config.plot_every == 0:
                plot_tracker(
                    (writer, model, iteration, config.plotting_functions), config.tracking_tool
                )

            if config.folder:
                if iteration % config.checkpoint_every == 0:
                    write_checkpoint(config.folder, model, optimizer, iteration)

            if iteration >= init_iter + config.max_iter:
                break

    # writing hyperparameters
    if config.hyperparams:
        log_tracker((writer, config.hyperparams, metrics), tracking_tool=config.tracking_tool)

    if config.log_model:
        log_model_tracker((writer, model, dataloader), tracking_tool=config.tracking_tool)

    # Final writing and checkpointing
    if config.folder:
        write_checkpoint(config.folder, model, optimizer, iteration)
    write_tracker((writer, loss, metrics, iteration), config.tracking_tool)

    # close tracker
    if config.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
        writer.close()
    elif config.tracking_tool == ExperimentTrackingTool.MLFLOW:
        writer.end_run()

    return model, optimizer
