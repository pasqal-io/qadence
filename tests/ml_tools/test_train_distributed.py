import os
import random
import time
import multiprocessing
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, Trainer, to_dataloader, DictDataLoader
from qadence.ml_tools.optimize_step import optimize_step


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def dictdataloader() -> DictDataLoader:
    data_configs: Dict[str, Dict[str, int]] = {
        "train": {"data_size": 50, "batch_size": 5},
        "val": {"data_size": 30, "batch_size": 5},
    }
    dls: Dict[str, DataLoader] = {}
    for name, config in data_configs.items():
        data_size: int = config["data_size"]
        batch_size: int = config["batch_size"]
        x = torch.rand(data_size, 1)
        y = torch.sin(x)
        dls[name] = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return DictDataLoader(dls)


def count_worker_processes(master_pid: int) -> int:
    """Count all worker processes spawned by the master process."""
    parent = psutil.Process(master_pid)
    children = parent.children(recursive=True)
    return len(children)


def normal_loss_fn(
    model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Picklable loss function for normal DataLoader."""
    x, y = data
    out = model(x)
    return nn.MSELoss()(out, y), {}


def dict_loss_fn(
    model: nn.Module, data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Picklable loss function for DictDataLoader."""
    losses = []
    for _, (x, y) in data_dict.items():
        out = model(x)
        losses.append(nn.MSELoss()(out, y))
    avg_loss = sum(losses) / len(losses)
    return avg_loss, {}


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_train_spawn(Basic: nn.Module, backend: str, device: str) -> None:
    """Test that Trainer.fit() correctly spawns multiple processes for normal DataLoader."""
    if device in ["gpu", "auto"] and not torch.cuda.is_available():
        pytest.skip("CUDA is not available for GPU/Auto mode.")
    if backend == "nccl" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for NCCL backend.")

    model: nn.Module = Basic
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.01)
    nprocs: int = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 60000))
    os.environ["WORLD_SIZE"] = str(nprocs)

    config: TrainConfig = TrainConfig(
        compute_setup=device,
        backend=backend,
        spawn=True,
        nprocs=nprocs,
        max_iter=5,
        print_every=1,
    )

    trainer: Trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        loss_fn=normal_loss_fn,
        train_dataloader=dataloader(batch_size=25),
        optimize_step=optimize_step,
    )

    master_pid: int = os.getpid()

    trainer.fit()

    time.sleep(5)  # Allow time for processes to spawn and settle
    worker_count: int = count_worker_processes(master_pid)

    assert (
        worker_count >= nprocs
    ), f"Expected at least {nprocs} worker processes, found {worker_count}"


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_train_spawn_dictdataloader(Basic: nn.Module, backend: str, device: str) -> None:
    """Test that Trainer.fit() correctly spawns multiple processes for DictDataLoader."""
    if device in ["gpu", "auto"] and not torch.cuda.is_available():
        pytest.skip("CUDA is not available for GPU/Auto mode.")
    if backend == "nccl" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for NCCL backend.")

    model: nn.Module = Basic
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=0.01)
    nprocs: int = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 60000))
    os.environ["WORLD_SIZE"] = str(nprocs)

    config: TrainConfig = TrainConfig(
        compute_setup=device,
        backend=backend,
        spawn=True,
        nprocs=nprocs,
        max_iter=5,
        print_every=1,
    )

    trainer: Trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        loss_fn=dict_loss_fn,
        train_dataloader=dictdataloader(),
        optimize_step=optimize_step,
    )

    master_pid: int = os.getpid()

    trainer.fit()

    time.sleep(5)  # Allow time for processes to spawn and settle
    worker_count: int = count_worker_processes(master_pid)

    assert (
        worker_count >= nprocs
    ), f"Expected at least {nprocs} worker processes, found {worker_count}"
