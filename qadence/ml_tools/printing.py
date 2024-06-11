from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


def print_metrics(loss: float | None, metrics: dict, iteration: int) -> None:
    msg = " ".join(
        [f"Iteration {iteration: >7} | Loss: {loss:.7f} -"]
        + [f"{k}: {v.item():.7f}" for k, v in metrics.items()]
    )
    print(msg)


def write_tensorboard(writer: SummaryWriter, loss: float, metrics: dict, iteration: int) -> None:
    writer.add_scalar("loss", loss, iteration)
    for key, arg in metrics.items():
        writer.add_scalar(key, arg, iteration)


def log_hyperparams(writer: SummaryWriter, hyperparams: dict, metrics: dict) -> None:
    writer.add_hparams(hyperparams, metrics)
