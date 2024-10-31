from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Union

from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.data import DictDataLoader
from qadence.types import ExperimentTrackingTool

logger = getLogger(__name__)

PlottingFunction = Callable[[Module, int], tuple[str, Figure]]
InputData = Union[Tensor, dict[str, Tensor]]


def print_metrics(loss: float | None, metrics: dict, iteration: int) -> None:
    msg = " ".join(
        [f"Iteration {iteration: >7} | Loss: {loss:.7f} -"]
        + [f"{k}: {v.item():.7f}" for k, v in metrics.items()]
    )
    print(msg)


def write_tensorboard(
    writer: SummaryWriter, loss: float = None, metrics: dict | None = None, iteration: int = 0
) -> None:
    metrics = metrics or dict()
    if loss is not None:
        writer.add_scalar("loss", loss, iteration)
    for key, arg in metrics.items():
        writer.add_scalar(key, arg, iteration)


def log_hyperparams_tensorboard(writer: SummaryWriter, hyperparams: dict, metrics: dict) -> None:
    writer.add_hparams(hyperparams, metrics)


def plot_tensorboard(
    writer: SummaryWriter,
    model: Module,
    iteration: int,
    plotting_functions: tuple[PlottingFunction],
) -> None:
    for pf in plotting_functions:
        descr, fig = pf(model, iteration)
        writer.add_figure(descr, fig, global_step=iteration)


def log_model_tensorboard(
    writer: SummaryWriter,
    model: Module,
    dataloader: Union[None, DataLoader, DictDataLoader],
) -> None:
    logger.warning("Model logging is not supported by tensorboard. No model will be logged.")


def write_mlflow(writer: Any, loss: float | None, metrics: dict, iteration: int) -> None:
    writer.log_metrics({"loss": float(loss)}, step=iteration)  # type: ignore
    writer.log_metrics(metrics, step=iteration)  # logs the single metrics


def log_hyperparams_mlflow(writer: Any, hyperparams: dict, metrics: dict) -> None:
    writer.log_params(hyperparams)  # type: ignore


def plot_mlflow(
    writer: Any,
    model: Module,
    iteration: int,
    plotting_functions: tuple[PlottingFunction],
) -> None:
    for pf in plotting_functions:
        descr, fig = pf(model, iteration)
        writer.log_figure(fig, descr)


def log_model_mlflow(
    writer: Any, model: Module, dataloader: DataLoader | DictDataLoader | None
) -> None:
    signature = None
    if dataloader is not None:
        xs: InputData
        xs, *_ = next(iter(dataloader))
        preds = model(xs)
        if isinstance(xs, Tensor):
            xs = xs.numpy()
            preds = preds.detach().numpy()
        elif isinstance(xs, dict):
            for key, val in xs.items():
                xs[key] = val.numpy()
            for key, val in preds.items():
                preds[key] = val.detach.numpy()

        try:
            from mlflow.models import infer_signature

            signature = infer_signature(xs, preds)
        except ImportError:
            logger.warning(
                "An MLFlow specific function has been called but MLFlow failed to import."
                "Please install MLFlow or adjust your code."
            )

    writer.pytorch.log_model(model, artifact_path="model", signature=signature)


TRACKER_MAPPING: dict[ExperimentTrackingTool, Callable[..., None]] = {
    ExperimentTrackingTool.TENSORBOARD: write_tensorboard,
    ExperimentTrackingTool.MLFLOW: write_mlflow,
}

LOGGER_MAPPING: dict[ExperimentTrackingTool, Callable[..., None]] = {
    ExperimentTrackingTool.TENSORBOARD: log_hyperparams_tensorboard,
    ExperimentTrackingTool.MLFLOW: log_hyperparams_mlflow,
}

PLOTTER_MAPPING: dict[ExperimentTrackingTool, Callable[..., None]] = {
    ExperimentTrackingTool.TENSORBOARD: plot_tensorboard,
    ExperimentTrackingTool.MLFLOW: plot_mlflow,
}

MODEL_LOGGER_MAPPING: dict[ExperimentTrackingTool, Callable[..., None]] = {
    ExperimentTrackingTool.TENSORBOARD: log_model_tensorboard,
    ExperimentTrackingTool.MLFLOW: log_model_mlflow,
}


def write_tracker(
    *args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return TRACKER_MAPPING[tracking_tool](*args)


def log_tracker(
    *args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return LOGGER_MAPPING[tracking_tool](*args)


def plot_tracker(
    *args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return PLOTTER_MAPPING[tracking_tool](*args)


def log_model_tracker(
    *args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return MODEL_LOGGER_MAPPING[tracking_tool](*args)
