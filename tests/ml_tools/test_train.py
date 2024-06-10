from __future__ import annotations

import os
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import DictDataLoader, TrainConfig, to_dataloader, train_with_grad
from qadence.ml_tools.models import TransformedModule
from qadence.models import QNN

torch.manual_seed(42)
np.random.seed(42)


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def dictdataloader(batch_size: int = 25, val: bool = False) -> DictDataLoader:
    x = torch.rand(batch_size, 1)
    y = torch.sin(x)
    dls = {
        "train" if val else "y1": to_dataloader(x, y, batch_size=batch_size, infinite=True),
        "val" if val else "y2": to_dataloader(x, y, batch_size=batch_size, infinite=True),
    }
    return DictDataLoader(dls)


def validation_criterion(
    current_validation_loss: float, current_best_validation_loss: float, val_epsilon: float
) -> bool:
    return current_validation_loss <= current_best_validation_loss - val_epsilon


def get_train_config_validation(
    tmp_path: Path, n_epochs: int, checkpoint_every: int, val_every: int
) -> TrainConfig:
    config = TrainConfig(
        folder=tmp_path,
        max_iter=n_epochs,
        print_every=10,
        checkpoint_every=checkpoint_every,
        write_every=100,
        val_every=val_every,
        checkpoint_best_only=True,
        validation_criterion=validation_criterion,
        val_epsilon=1e-5,
    )
    return config


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

    n_epochs = 100
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
def test_train_tensor_tuple(Basic: torch.nn.Module, BasicQNN: QNN) -> None:
    for cls, dtype in [(Basic, torch.float32), (BasicQNN, torch.complex64)]:
        model = TransformedModule(cls, 1, 1, *[torch.nn.Parameter(t) for t in torch.rand(4)])
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
            max_iter=n_epochs,
            checkpoint_every=100,
            write_every=100,
            batch_size=batch_size,
        )
        data = to_dataloader(x, y, batch_size=batch_size, infinite=True)
        model, _ = train_with_grad(model, data, optimizer, config, loss_fn=loss_fn, dtype=dtype)
        assert next(cnt) == n_epochs

        x = torch.rand(5, 1, dtype=torch.float32)
        assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)


@pytest.mark.flaky(max_runs=10)
def test_fit_sin_adjoint(BasicAdjointQNN: torch.nn.Module) -> None:
    model = BasicAdjointQNN
    batch_size = 5
    n_epochs = 200
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        x_train = torch.rand(batch_size, 1)
        y_train = torch.sin(x_train)
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    x_test = torch.rand(1, 1)
    assert torch.allclose(torch.sin(x_test), model(x_test), rtol=1e-1, atol=1e-1)


def test_train_dataloader_val_check_and_non_dict_dataloader(
    tmp_path: Path, Basic: torch.nn.Module
) -> None:
    data = dataloader()
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data["y1"][0], data["y1"][1]
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    with pytest.raises(ValueError) as exc_info:
        train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert (
        "If `config.val_every` is provided as an integer, dataloader must"
        "be an instance of `DictDataLoader`" in exc_info.exconly()
    )


def test_train_dataloader_val_check_incorrect_keys(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    data = dictdataloader(batch_size=batch_size, val=False)  # Passing val=False to raise an error.
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data[0], data[1]
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    with pytest.raises(ValueError) as exc_info:
        train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert (
        "If `config.val_every` is provided as an integer, the dictdataloader"
        "must have `train` and `val` keys to access the respective dataloaders."
        in exc_info.exconly()
    )


def test_train_dictdataloader_checkpoint_best_only(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    data = dictdataloader(batch_size=batch_size, val=True)
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data[0], data[1]
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
    assert next(cnt) == n_epochs + n_epochs // val_every

    files = [f for f in os.listdir(tmp_path) if f.endswith(".pt") and "model" in f]
    # Ideally it can be ensured if the (only) saved checkpoint is indeed the best,
    # but that is time-consuming since training must be run twice for comparison.
    # The below check may be plausible enough.
    assert len(files) == 1  # Since only the best checkpoint must be stored.
