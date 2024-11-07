from __future__ import annotations

import os
from itertools import count
from pathlib import Path

import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from qadence import QNN, QuantumModel
from qadence.ml_tools import (
    TrainConfig,
    Trainer,
    load_checkpoint,
)
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.parameters import get_parameters, set_parameters
from qadence.ml_tools.utils import rand_featureparameters


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def write_legacy_checkpoint(
    folder: Path,
    model: torch.Module,
    optimizer: Optimizer | NGOptimizer,
    iteration: int | str,
) -> None:
    iteration_substring = f"{iteration:03n}" if isinstance(iteration, int) else iteration
    model_checkpoint_name: str = f"model_{type(model).__name__}_ckpt_{iteration_substring}.pt"
    opt_checkpoint_name: str = f"opt_{type(optimizer).__name__}_ckpt_{iteration_substring}.pt"
    d = (
        model._to_dict(save_params=True)
        if isinstance(model, (QNN, QuantumModel))
        else model.state_dict()
    )
    torch.save((iteration, d), folder / model_checkpoint_name)
    if isinstance(optimizer, Optimizer):
        torch.save(
            (iteration, type(optimizer), optimizer.state_dict()), folder / opt_checkpoint_name
        )
    elif isinstance(optimizer, NGOptimizer):
        optimizer.dump(folder / opt_checkpoint_name)


def test_basic_save_load_ckpts(Basic: torch.nn.Module, tmp_path: Path) -> None:
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

    config = TrainConfig(folder=tmp_path, max_iter=1, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        model, _ = trainer.fit()
    ps0 = get_parameters(model)
    set_parameters(model, torch.ones(len(get_parameters(model))))
    # write_checkpoint(tmp_path, model, optimizer, 1)
    # check that saved model has ones
    model, _, _ = load_checkpoint(trainer.config._log_folder, model, optimizer)
    ps1 = get_parameters(model)
    assert torch.allclose(ps0, ps1)


def test_random_basic_qm_save_load_ckpts(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQuantumModel
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation({}).squeeze(dim=0)
        loss = criterion(out, torch.rand(1))
        return loss, {}

    config = TrainConfig(folder=tmp_path, max_iter=10, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        model, optimizer = trainer.fit()
    ps0 = get_parameters(model)
    ev0 = model.expectation({})
    # Modify model's parameters
    set_parameters(model, torch.ones(len(ps0)))

    model, optimizer, _ = load_checkpoint(trainer.config._log_folder, model, optimizer)
    ps1 = get_parameters(model)
    ev1 = model.expectation({})

    assert not torch.all(torch.isnan(ev1))
    assert torch.allclose(ps0, ps1)
    assert torch.allclose(ev0, ev1)

    loaded_model, optimizer, _ = load_checkpoint(
        trainer.config._log_folder,
        BasicQuantumModel,
        optimizer,
        "model_QuantumModel_ckpt_009_device_cpu.pt",
        "opt_Adam_ckpt_006_device_cpu.pt",
    )
    assert not torch.all(torch.isnan(loaded_model.expectation({})))


def test_check_ckpts_exist(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQuantumModel
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation({}).squeeze(dim=0)
        loss = criterion(out, torch.rand(1))
        return loss, {}

    config = TrainConfig(folder=tmp_path, max_iter=10, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        trainer.fit()
    ckpts = [
        trainer.config._log_folder / Path(f"model_QuantumModel_ckpt_00{i}_device_cpu.pt")
        for i in range(1, 9)
    ]
    assert all(os.path.isfile(ckpt) for ckpt in ckpts)
    for ckpt in ckpts:
        loaded_model, optimizer, _ = load_checkpoint(
            tmp_path, BasicQuantumModel, optimizer, ckpt, ""
        )
        assert torch.allclose(loaded_model.expectation({}), model.expectation({}))


def test_random_basic_qnn_save_load_ckpts(BasicQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQNN
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    inputs = rand_featureparameters(model, 1)

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation(inputs).squeeze(dim=0)
        loss = criterion(out, torch.rand(1))
        return loss, {}

    config = TrainConfig(folder=tmp_path, max_iter=10, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        trainer.fit()
    ps0 = get_parameters(model)
    ev0 = model.expectation(inputs)

    # Modify model's parameters
    set_parameters(model, torch.ones(len(ps0)))

    model, optimizer, _ = load_checkpoint(trainer.config._log_folder, model, optimizer)
    ps1 = get_parameters(model)
    ev1 = model.expectation(inputs)

    assert torch.allclose(ps0, ps1)
    assert torch.allclose(ev0, ev1)

    # Test loading the last ckpt by name
    loaded_model, optimizer, _ = load_checkpoint(
        tmp_path,
        BasicQNN,
        optimizer,
        "model_QNN_ckpt_003_device_cpu.pt",
        "opt_Adam_ckpt_006_device_cpu.pt",
    )
    assert not torch.all(torch.isnan(loaded_model.expectation(inputs)))


def test_check_qnn_ckpts_exist(BasicQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQNN
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    inputs = rand_featureparameters(model, 1)

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation(inputs).squeeze(dim=0)
        loss = criterion(out, torch.rand(1))
        return loss, {}

    config = TrainConfig(folder=tmp_path, max_iter=10, checkpoint_every=1, write_every=1)
    trainer = Trainer(model, optimizer, config, loss_fn, data)
    with trainer.enable_grad_opt():
        trainer.fit()
    ckpts = [
        trainer.config._log_folder / Path(f"model_QNN_ckpt_00{i}_device_cpu.pt")
        for i in range(1, 9)
    ]
    assert all(os.path.isfile(ckpt) for ckpt in ckpts)
    for ckpt in ckpts:
        loaded_model, optimizer, _ = load_checkpoint(tmp_path, BasicQNN, optimizer, ckpt, "")
        assert torch.allclose(loaded_model.expectation(inputs), model.expectation(inputs))


def test_basic_qm_save_load_legacy_ckpts(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    model = BasicQuantumModel
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    ps0 = get_parameters(model)
    write_legacy_checkpoint(tmp_path, model, optimizer, 1)

    # Modify model's parameters to make sure we are loading the saved ckpt
    set_parameters(model, torch.ones(len(ps0)))

    loaded_model, optimizer, _ = load_checkpoint(tmp_path, model, optimizer)
    ps1 = get_parameters(loaded_model)
    assert not torch.all(torch.isnan(loaded_model.expectation({})))
    assert torch.allclose(ps0, ps1)


def test_basic_qnn_save_load_legacy_ckpts(BasicQNN: QNN, tmp_path: Path) -> None:
    model = BasicQNN
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    inputs = rand_featureparameters(model, 1)

    ps0 = get_parameters(model)
    write_legacy_checkpoint(tmp_path, model, optimizer, 1)

    # Modify model's parameters to make sure we are loading the saved ckpt
    set_parameters(model, torch.ones(len(ps0)))

    loaded_model, optimizer, _ = load_checkpoint(tmp_path, model, optimizer)
    ps1 = get_parameters(loaded_model)
    assert not torch.all(torch.isnan(model.expectation(inputs)))
    assert torch.allclose(ps0, ps1)
