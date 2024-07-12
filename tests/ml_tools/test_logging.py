from __future__ import annotations

import os
from itertools import count
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, train_with_grad
from qadence.ml_tools.data import to_dataloader
from qadence.models import QuantumModel
from qadence.types import ExperimentTrackingTool


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def test_hyperparams_logging_mlflow(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQuantumModel
    cnt = count()
    criterion = torch.nn.MSELoss()
    hyperparams = {"max_iter": int(10), "lr": 0.1}
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation({})
        loss = criterion(out, torch.rand(1))
        return loss, {}

    config = TrainConfig(
        folder=tmp_path,
        max_iter=hyperparams["max_iter"],  # type: ignore
        checkpoint_every=1,
        write_every=1,
        hyperparams=hyperparams,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

    mlflow_config = config.mlflow_config
    experiment_id = mlflow_config.run.info.experiment_id
    run_id = mlflow_config.run.info.run_id

    hyperparams_files = [
        Path(f"mlruns/{experiment_id}/{run_id}/params/{key}") for key in hyperparams.keys()
    ]
    assert all([os.path.isfile(hf) for hf in hyperparams_files])


def test_plotting() -> None:
    pass
