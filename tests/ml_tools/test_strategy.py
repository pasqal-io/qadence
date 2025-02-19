from __future__ import annotations

import os
from typing import Any
import torch.distributed as dist

from qadence.ml_tools.train_utils.strategy import DistributionStrategy


def test_detect_strategy_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    ds = DistributionStrategy(compute_setup="cpu")
    ds.spawn = False  # even if spawn is False, presence of LOCAL_RANK implies default
    strategy = ds.detect_strategy()
    assert strategy == "default"
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_detect_strategy_torchrun(monkeypatch: Any) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "dummy")
    ds = DistributionStrategy(compute_setup="cpu")
    ds.spawn = False
    strategy = ds.detect_strategy()
    assert strategy == "torchrun"
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)


def test_detect_strategy_none(monkeypatch: Any) -> None:
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    ds = DistributionStrategy(compute_setup="cpu")
    ds.spawn = False
    strategy = ds.detect_strategy()
    assert strategy == "none"


def test_set_cluster_variables(monkeypatch: Any) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node1,node2")
    ds = DistributionStrategy(compute_setup="cpu")
    ds._set_cluster_variables()
    assert ds.job_id == 12345
    assert ds.num_nodes == 2
    assert ds.node_list == "node1,node2"


def test_get_master_addr_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASTER_ADDR", "1.2.3.4")
    ds = DistributionStrategy(compute_setup="cpu")
    ds.strategy = "none"
    addr = ds._get_master_addr()
    assert addr == "1.2.3.4"
    monkeypatch.delenv("MASTER_ADDR", raising=False)


def test_get_master_port_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASTER_PORT", "23456")
    ds = DistributionStrategy(compute_setup="cpu")
    ds.strategy = "none"
    port = ds._get_master_port()
    assert port == "23456"
    monkeypatch.delenv("MASTER_PORT", raising=False)


def test_get_master_port_default(monkeypatch: Any) -> None:
    monkeypatch.delenv("MASTER_PORT", raising=False)
    ds = DistributionStrategy(compute_setup="cpu")
    ds.strategy = "default"
    ds.job_id = 12000
    port = ds._get_master_port()
    expected_port = str(int(12000 + 12000 % 5000))
    assert port == expected_port


def test_setup_environment(monkeypatch: Any) -> None:
    # Set necessary environment variables for SLURM-like setup.
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "test_node")
    ds = DistributionStrategy(compute_setup="cpu")
    ds.strategy = "default"
    ds.spawn = False
    ds.nprocs = 4
    rank, world_size, local_rank = ds.setup_environment(1)
    assert isinstance(rank, int)
    assert isinstance(world_size, int)

    # In CPU mode, local_rank is not defined (None)
    assert local_rank is None

    # Environment variables should have been set.
    assert os.environ["RANK"] == str(rank)
    assert os.environ["WORLD_SIZE"] == str(world_size)
    assert "LOCAL_RANK" in os.environ


def test_setup_process() -> None:
    ds = DistributionStrategy(compute_setup="cpu")
    ds.strategy = "default"
    ds.spawn = False
    ds.nprocs = 4
    rank, world_size, local_rank, device = ds.setup_process(0, 4)
    assert device == "cpu"
    assert isinstance(rank, int)
    assert isinstance(world_size, int)
    # In CPU mode, local_rank should be None.
    assert local_rank is None


def test_start_process_group(monkeypatch: Any) -> None:
    ds = DistributionStrategy(compute_setup="cpu")
    ds.world_size = 2
    ds.rank = 0
    ds.master_addr = "localhost"
    ds.master_port = "12345"
    init_called = False
    barrier_called = False

    def fake_init_process_group(backend: str, rank: int, world_size: int) -> None:
        nonlocal init_called
        init_called = True

    def fake_barrier() -> None:
        nonlocal barrier_called
        barrier_called = True

    monkeypatch.setattr(dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(dist, "barrier", fake_barrier)
    ds.start_process_group()
    assert init_called is True
    assert barrier_called is True


def test_cleanup_process_group(monkeypatch: Any) -> None:
    ds = DistributionStrategy(compute_setup="cpu")
    ds.rank = 0
    destroy_called = False

    def fake_destroy() -> None:
        nonlocal destroy_called
        destroy_called = True

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", fake_destroy)
    ds.cleanup_process_group()
    assert destroy_called is True
