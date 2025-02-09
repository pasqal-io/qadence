from __future__ import annotations

import os
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import QNN, DictDataLoader, TrainConfig, Trainer, to_dataloader

torch.manual_seed(42)
np.random.seed(42)


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def dictdataloader(data_configs: dict[str, dict[str, int]]) -> DictDataLoader:
    dls = {}
    for name, config in data_configs.items():
        data_size = config["data_size"]
        batch_size = config["batch_size"]
        x = torch.rand(data_size, 1)
        y = torch.sin(x)
        dls[name] = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return DictDataLoader(dls)


def train_val_dataloaders(batch_size: int = 25) -> tuple:
    x = torch.rand(batch_size, 1)
    y = torch.sin(x)
    train_dataloader = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    val_dataloader = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return train_dataloader, val_dataloader


def validation_criterion(
    current_validation_loss: float, current_best_validation_loss: float, val_epsilon: float
) -> bool:
    return current_validation_loss <= current_best_validation_loss - val_epsilon


def get_train_config_validation(
    tmp_path: Path, n_epochs: int, checkpoint_every: int, val_every: int
) -> TrainConfig:
    config = TrainConfig(
        root_folder=tmp_path,
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
        x, y = data
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    n_epochs = 100
    config = TrainConfig(
        root_folder=tmp_path, max_iter=n_epochs, checkpoint_every=100, write_every=100
    )
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        trainer.fit()
    assert next(cnt) == (n_epochs + 1)

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
        root_folder=tmp_path,
        max_iter=n_epochs,
        print_every=5,
        checkpoint_every=100,
        write_every=100,
    )
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        trainer.fit()
    assert next(cnt) == (n_epochs + 1)

    out = model()
    assert torch.allclose(out, torch.zeros(1), atol=1e-2, rtol=1e-2)


@pytest.mark.flaky(max_runs=10)
def test_train_val(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    train_data, val_data = train_val_dataloaders(batch_size=batch_size)
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data
        l1 = criterion(model(x1), y1)
        return l1, {}

    n_epochs = 100
    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=n_epochs,
        print_every=10,
        checkpoint_every=100,
        write_every=100,
    )
    trainer = Trainer(
        model, optimizer, config, loss_fn, train_dataloader=train_data, val_dataloader=val_data
    )
    with trainer.enable_grad_opt():
        trainer.fit()
    assert next(cnt) == (n_epochs + 1)

    x = torch.rand(5, 1)
    assert torch.allclose(torch.sin(x), model(x), rtol=1e-1, atol=1e-1)


@pytest.mark.flaky(max_runs=10)
def test_train_tensor_tuple(Basic: torch.nn.Module, BasicQNN: QNN) -> None:
    for cls, dtype in [(Basic, torch.float32), (BasicQNN, torch.complex64)]:
        model = cls
        batch_size = 25
        x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
        y = torch.sin(x)
        model = model.to(
            torch.float32
        )  # BasicQNN might have float64, and Adam behaves weirdly with mixed precision

        cnt = count()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
            next(cnt)
            x, y = data
            out = model(x)
            loss = criterion(out, y)
            return loss, {}

        n_epochs = 100
        config = TrainConfig(
            max_iter=n_epochs,
            checkpoint_every=100,
            write_every=100,
            batch_size=batch_size,
            dtype=dtype,
        )
        data = to_dataloader(x, y, batch_size=batch_size, infinite=True)
        trainer = Trainer(model, optimizer, config, loss_fn, data)
        with trainer.enable_grad_opt():
            model, _ = trainer.fit()
        assert next(cnt) == (n_epochs + 1)

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
        x1, y1 = data
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    with pytest.raises(ValueError) as exc_info:
        trainer = Trainer(model, optimizer, config, loss_fn, data)
        with trainer.enable_grad_opt():
            trainer.fit()
    assert (
        "If `config.val_every` is provided as an integer > 0, validation_dataloader"
        "must be an instance of `DataLoader` or `DictDataLoader`." in exc_info.exconly()
    )


def test_train_dataloader_val_check_incorrect_keys(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    train_data, _ = train_val_dataloaders(batch_size=batch_size)
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    with pytest.raises(ValueError) as exc_info:
        trainer = Trainer(
            model, optimizer, config, loss_fn, train_dataloader=train_data, val_dataloader=None
        )
        with trainer.enable_grad_opt():
            trainer.fit()
    assert (
        "If `config.val_every` is provided as an integer > 0, validation_dataloader"
        "must be an instance of `DataLoader` or `DictDataLoader`." in exc_info.exconly()
    )


def test_train_val_checkpoint_best_only(tmp_path: Path, Basic: torch.nn.Module) -> None:
    batch_size = 25
    train_data, val_data = train_val_dataloaders(batch_size=batch_size)
    model = Basic

    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        x1, y1 = data
        loss = criterion(model(x1), y1)
        return loss, {}

    n_epochs = 100
    checkpoint_every = 20
    val_every = 10

    config = get_train_config_validation(tmp_path, n_epochs, checkpoint_every, val_every)
    trainer = Trainer(
        model, optimizer, config, loss_fn, train_dataloader=train_data, val_dataloader=val_data
    )
    with trainer.enable_grad_opt():
        trainer.fit()
    assert next(cnt) == 2 + n_epochs + (n_epochs // val_every) + 1  # 1 for intial round 0 run

    files = [f for f in os.listdir(trainer.config.log_folder) if f.endswith(".pt") and "model" in f]
    # Ideally it can be ensured if the (only) saved checkpoint is indeed the best,
    # but that is time-consuming since training must be run twice for comparison.
    # The below check may be plausible enough.
    assert len(files) == 1  # Since only the best checkpoint must be stored.


def test_dict_dataloader_with_trainer(tmp_path: Path, Basic: torch.nn.Module) -> None:
    data_configs = {
        "dataset1": {"data_size": 30, "batch_size": 5},
        "dataset2": {"data_size": 50, "batch_size": 10},
    }
    dict_loader = dictdataloader(data_configs)

    # Define the model, loss function, optimizer
    model = Basic
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(model: torch.nn.Module, data: dict) -> tuple[torch.Tensor, dict]:
        losses = []
        for key, (x, y) in data.items():
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss)
        total_loss = sum(losses) / len(losses)
        return total_loss, {}

    config = TrainConfig(
        root_folder=tmp_path,
        max_iter=50,
        checkpoint_every=10,
        write_every=10,
    )

    trainer = Trainer(model, optimizer, config, loss_fn, dict_loader)
    with trainer.enable_grad_opt():
        trainer.fit()

    x = torch.rand(5, 1)
    for key in dict_loader.dataloaders.keys():
        y_pred = model(x)
        assert y_pred.shape == (5, 1)


def test_dict_dataloader() -> None:
    data_configs = {
        "dataset1": {"data_size": 20, "batch_size": 5},
        "dataset2": {"data_size": 40, "batch_size": 10},
    }
    ddl = dictdataloader(data_configs)
    assert set(ddl.dataloaders.keys()) == {"dataset1", "dataset2"}

    batch = next(iter(ddl))
    assert batch["dataset1"][0].shape == (5, 1)
    assert batch["dataset2"][0].shape == (10, 1)
    for key, (x, y) in batch.items():
        assert x.shape == y.shape
