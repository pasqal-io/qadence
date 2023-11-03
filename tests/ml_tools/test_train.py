from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from qadence.ml_tools import DictDataLoader, TrainConfig, train_with_grad, to_dataloader
from qadence.ml_tools.models import TransformedModule
from qadence.models import QNN

torch.manual_seed(42)
np.random.seed(42)


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def dictdataloader(batch_size: int = 25) -> DictDataLoader:
    x = torch.rand(batch_size, 1)
    y = torch.sin(x)
    dls = {
        "y1": to_dataloader(x, y, batch_size=batch_size, infinite=True),
        "y2": to_dataloader(x, y, batch_size=batch_size, infinite=True)
    }
    return DictDataLoader(dls)


def FMdictdataloader(param_name: str = "phi", n_qubits: int = 2) -> DictDataLoader:
    batch_size = 1
    x = torch.rand(batch_size, 1)
    y = torch.sin(x)
    return DictDataLoader({param_name: to_dataloader(x, y, batch_size=batch_size, infinite=True)})


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

    n_epochs = 100
    config = TrainConfig(folder=tmp_path, max_iter=n_epochs, checkpoint_every=100, write_every=100)
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert next(cnt) == n_epochs

    x = torch.rand(5, 1)
    assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)


def test_train_dataloader_no_data(tmp_path: Path, BasicNoInput: torch.nn.Module) -> None:
    data = None
    model = BasicNoInput

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    def loss_fn(model: torch.nn.Module, xs: Any = None) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model()
        loss = criterion(out, torch.tensor([0.0]))
        return loss, {}

    n_epochs = 50
    config = TrainConfig(
        folder=tmp_path,
        max_iter=n_epochs,
        print_every=5,
        checkpoint_every=100,
        write_every=100,
    )
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert next(cnt) == n_epochs

    out = model()
    assert torch.allclose(out, torch.zeros(1), atol=1e-2, rtol=1e-2)


@pytest.mark.flaky(max_runs=10)
def test_train_dictdataloader(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    data = dictdataloader(batch_size=batch_size)
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data["y1"][0], data["y1"][1]
        x2, y2 = data["y2"][0], data["y2"][1]
        l1 = criterion(model(x1), y1)
        l2 = criterion(model(x2), y2)
        return l1 + l2, {}

    n_epochs = 200
    config = TrainConfig(
        folder=tmp_path, max_iter=n_epochs, print_every=10, checkpoint_every=100, write_every=100
    )
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert next(cnt) == n_epochs

    x = torch.rand(5, 1)
    assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)


@pytest.mark.slow
@pytest.mark.flaky(max_runs=10)
def test_modules_save_load(BasicQNN: QNN, BasicTransformedModule: TransformedModule) -> None:
    data = FMdictdataloader()
    for _m in [BasicQNN, BasicTransformedModule]:
        model: torch.nn.Module = _m
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
            x = torch.rand(1)
            y = torch.sin(x)
            l1 = criterion(model(x), y)
            return l1, {}

        n_epochs = 200
        config = TrainConfig(
            max_iter=n_epochs, print_every=10, checkpoint_every=500, write_every=500
        )
        model, optimizer = train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
        x = torch.rand(1)
        assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)


@pytest.mark.flaky(max_runs=10)
def test_train_tensor_tuple(tmp_path: Path, Basic: torch.nn.Module) -> None:
    model = Basic
    batch_size = 25
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x, y = data[0], data[1]
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    n_epochs = 100
    config = TrainConfig(
        folder=tmp_path,
        max_iter=n_epochs,
        checkpoint_every=100,
        write_every=100,
        batch_size=batch_size,
    )
    data = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert next(cnt) == n_epochs

    x = torch.rand(5, 1)
    assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)
