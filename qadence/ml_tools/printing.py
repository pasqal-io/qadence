from __future__ import annotations

from typing import Any

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


def write_mlflow(writer: Any, loss: float | None, metrics: dict, iteration: int) -> None:
    writer.log_metrics({"loss": float(loss)}, step=iteration)  # type: ignore
    writer.log_metrics(metrics, step=iteration)  # logs the single metrics


def log_hyperparams_mlflow(writer: Any, hyperparams: dict, metrics: dict) -> None:
    writer.log_params(hyperparams)


TRACKER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: write_tensorboard,
    ExperimentTrackingTool.MLFLOW: write_mlflow,
}

LOGGER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: log_hyperparams_tensorboard,
    ExperimentTrackingTool.MLFLOW: log_hyperparams_mlflow,
}


def write_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return TRACKER_MAPPING[tracking_tool](*args)


def log_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return LOGGER_MAPPING[tracking_tool](*args)
