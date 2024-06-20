from __future__ import annotations

from typing import Any, Callable

from matplotlib.figure import Figure
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from qadence.types import ExperimentTrackingTool


def print_metrics(loss: float | None, metrics: dict, iteration: int) -> None:
    msg = " ".join(
        [f"Iteration {iteration: >7} | Loss: {loss:.7f} -"]
        + [f"{k}: {v.item():.7f}" for k, v in metrics.items()]
    )
    print(msg)


def write_tensorboard(
    writer: SummaryWriter, loss: float | None, metrics: dict, iteration: int
) -> None:
    writer.add_scalar("loss", loss, iteration)
    for key, arg in metrics.items():
        writer.add_scalar(key, arg, iteration)


def log_hyperparams_tensorboard(writer: SummaryWriter, hyperparams: dict, metrics: dict) -> None:
    writer.add_hparams(hyperparams, metrics)


def plot_tensorboard(
    writer: SummaryWriter, iteration: int, plotting_functions: tuple[Callable]
) -> None:
    raise NotImplementedError("Plot logging with tensorboard is not implemented")


def write_mlflow(writer: Any, loss: float | None, metrics: dict, iteration: int) -> None:
    writer.log_metrics({"loss": float(loss)}, step=iteration)  # type: ignore
    writer.log_metrics(metrics, step=iteration)  # logs the single metrics


def log_hyperparams_mlflow(writer: Any, hyperparams: dict, metrics: dict) -> None:
    writer.log_params(hyperparams)  # type: ignore


def plot_mlflow(
    writer: SummaryWriter,
    model: Module,
    iteration: int,
    plotting_functions: tuple[Callable[[Module, int], tuple[str, Figure]]],
) -> None:
    for pf in plotting_functions:
        descr, fig = pf(model, iteration)
        writer.log_figure(fig, descr)


TRACKER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: write_tensorboard,
    ExperimentTrackingTool.MLFLOW: write_mlflow,
}

LOGGER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: log_hyperparams_tensorboard,
    ExperimentTrackingTool.MLFLOW: log_hyperparams_mlflow,
}

PLOTTER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: plot_tensorboard,
    ExperimentTrackingTool.MLFLOW: plot_mlflow,
}


def write_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return TRACKER_MAPPING[tracking_tool](*args)


def log_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return LOGGER_MAPPING[tracking_tool](*args)


def plot_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return PLOTTER_MAPPING[tracking_tool](*args)  # type: ignore
