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


def log_hyperparams(writer: SummaryWriter, hyperparams: dict, metrics: dict) -> None:
    writer.add_hparams(hyperparams, metrics)


def write_mflow(writer: Any, loss: float | None, metrics: dict, iteration: int) -> None:
    # TODO for giorgio
    # if we use the pytorch.autolog, we can just open a context
    pass


TRACKER_MAPPING = {
    ExperimentTrackingTool.TENSORBOARD: write_tensorboard,
    ExperimentTrackingTool.MLFLOW: write_mflow,
}


def write_tracker(
    args: Any, tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
) -> None:
    return TRACKER_MAPPING[tracking_tool](*args)
