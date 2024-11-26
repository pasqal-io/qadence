from __future__ import annotations

import random
from itertools import count
from pathlib import Path

import nevergrad as ng
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, Trainer, num_parameters, to_dataloader

# ensure reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


@pytest.mark.flaky(max_runs=10)
def test_train_dataloader_default(tmp_path: Path, Basic: torch.nn.Module) -> None:
    data = dataloader()
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x, y = data[0], data[1]
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    n_epochs = 500
    config = TrainConfig(
        root_folder=tmp_path, max_iter=n_epochs, checkpoint_every=100, write_every=100
    )

    optimizer = ng.optimizers.NGOpt(budget=config.max_iter, parametrization=num_parameters(model))

    trainer = Trainer(model, None, config=config, loss_fn=loss_fn, train_dataloader=data)
    with trainer.disable_grad_opt(optimizer):
        trainer.fit()
    assert next(cnt) == n_epochs + 1

    x = torch.rand(5, 1)
    assert torch.allclose(torch.cos(x), model(x), rtol=1e-1, atol=1e-1)
