from __future__ import annotations

import os
from typing import Any
import torch.distributed as dist

from qadence.ml_tools.train_utils.distribution import Distributor
from qadence.ml_tools.train_utils.execution import detect_execution
from qadence.types import ExecutionType


def test_detect_execution_default(monkeypatch: Any) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
    assert ExecutionType.DEFAULT == ds.execution_type
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_set_cluster_variables(monkeypatch: Any) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node1,node2")
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
    ds.execution._set_cluster_variables()
    assert ds.execution.job_id == "12345"
    assert ds.execution.num_nodes == 2
    assert ds.execution.node_list == "node1,node2"


def test_get_master_addr_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASTER_ADDR", "1.2.3.4")
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
    addr = ds.execution.get_master_addr()
    assert addr == "1.2.3.4"
    monkeypatch.delenv("MASTER_ADDR", raising=False)


def test_get_master_port_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASTER_PORT", "23456")
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
    port = ds.execution.get_master_port()
    assert port == "23456"
    monkeypatch.delenv("MASTER_PORT", raising=False)


def test_setup_environment(monkeypatch: Any) -> None:
    # Set necessary environment variables for SLURM-like setup.
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", "test_node")
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
    ds.execution_type = ExecutionType.DEFAULT
    values_dict = ds.setup_process_rank_environment(0)
    local_rank, world_size, rank = (
        values_dict["LOCAL_RANK"],
        values_dict["WORLD_SIZE"],
        values_dict["RANK"],
    )
    assert isinstance(rank, int)
    assert isinstance(world_size, int)

    # In CPU mode, local_rank is not defined (None)
    assert local_rank is None

    # Environment variables should have been set.
    assert os.environ["RANK"] == str(rank)
    assert os.environ["WORLD_SIZE"] == str(world_size)
    assert "LOCAL_RANK" in os.environ


def test_start_process_group(monkeypatch: Any) -> None:
    ds = Distributor(compute_setup="cpu", log_setup="cpu", nprocs=1, backend="gloo")
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
    ds = Distributor(compute_setup="cpu", nprocs=1, backend="gloo", log_setup="cpu")
    ds.rank = 0
    destroy_called = False

    def fake_destroy() -> None:
        nonlocal destroy_called
        destroy_called = True

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", fake_destroy)
    ds.finalize()
    assert destroy_called is True
