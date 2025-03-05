import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from unittest.mock import patch
from typing import Any, Dict, List

from qadence.ml_tools.train_utils.accelerator import Accelerator
from qadence.ml_tools.data import DictDataLoader, to_dataloader
from qadence.types import ExecutionType


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def get_dummy_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.SGD(model.parameters(), lr=0.1)


def get_dummy_dataloader(batch_size: int = 4) -> DataLoader:
    x = torch.randn(20, 10)
    y = torch.randn(20, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()


@pytest.fixture
def dummy_optimizer(dummy_model: DummyModel) -> optim.Optimizer:
    return get_dummy_optimizer(dummy_model)


@pytest.fixture
def dummy_dataloader() -> DataLoader:
    return get_dummy_dataloader(batch_size=4)


@pytest.fixture
def dict_dataloader() -> DictDataLoader:
    data_configs: Dict[str, Dict[str, int]] = {
        "dataset1": {"data_size": 30, "batch_size": 5},
        "dataset2": {"data_size": 50, "batch_size": 10},
    }
    dataloaders: Dict[str, DataLoader] = {}
    for name, cfg in data_configs.items():
        x = torch.randn(cfg["data_size"], 10)
        y = torch.randn(cfg["data_size"], 1)
        dataloaders[name] = to_dataloader(x, y, batch_size=cfg["batch_size"])
    return DictDataLoader(dataloaders)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ml_tools")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


def test_prepare_model_without_ddp() -> None:
    accelerator = Accelerator(nprocs=1, compute_setup="cpu")
    accelerator.world_size = 1
    accelerator.execution.device = "cpu"
    model = DummyModel()
    prepared_model = accelerator._prepare_model(model)
    assert not isinstance(prepared_model, DDP)
    for param in prepared_model.parameters():
        assert param.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prepare_model_with_ddp_cuda() -> None:
    accelerator = Accelerator(nprocs=2, compute_setup="gpu")
    accelerator.world_size = 2
    accelerator.local_rank = 0
    accelerator.execution.device = "cuda:0"
    model = DummyModel()
    prepared_model = accelerator._prepare_model(model)
    assert isinstance(prepared_model, DDP)
    for param in prepared_model.parameters():
        assert param.device.type == "cuda"


def test_prepare_optimizer() -> None:
    accelerator = Accelerator(nprocs=1)
    model = DummyModel()
    optimizer = get_dummy_optimizer(model)
    prepared_optimizer = accelerator._prepare_optimizer(optimizer)
    assert prepared_optimizer is optimizer


@pytest.mark.parametrize("world_size", [1, 2])
def test_prepare_data_without_distributed(world_size: int) -> None:
    accelerator = Accelerator(nprocs=world_size)
    accelerator.world_size = world_size
    if world_size == 2:
        accelerator.local_rank = 0
    dl = get_dummy_dataloader()
    prepared_dl = accelerator._prepare_data(dl)
    if world_size == 1:
        assert prepared_dl is dl
    else:
        assert isinstance(prepared_dl.sampler, DistributedSampler)  # type: ignore[union-attr]


def test_prepare_dict_dataloader(dict_dataloader: DictDataLoader) -> None:
    accelerator = Accelerator(nprocs=1)
    accelerator.world_size = 2
    accelerator.local_rank = 0
    prepared_dl = accelerator._prepare_data(dict_dataloader)
    assert isinstance(prepared_dl, DictDataLoader)
    for name, dl in prepared_dl.dataloaders.items():
        assert isinstance(dl.sampler, DistributedSampler)


def test_prepare_function(
    dummy_model: DummyModel, dummy_optimizer: optim.Optimizer, dummy_dataloader: DataLoader
) -> None:
    accelerator = Accelerator(nprocs=1)
    model_prepped, opt_prepped, dl_prepped = accelerator.prepare(
        dummy_model, dummy_optimizer, dummy_dataloader
    )
    assert isinstance(model_prepped, nn.Module)
    assert isinstance(opt_prepped, optim.Optimizer)
    assert isinstance(dl_prepped, DataLoader)


def dummy_worker(run_processes: mp.Value) -> None:
    with run_processes.get_lock():
        run_processes.value += 1


@pytest.mark.skip(reason="Spawning processes inside github CI causes problems")
@pytest.mark.parametrize("nprocs", [3])
def test_spawn_multiple_methods(nprocs: int) -> None:
    run_processes = mp.Value("i", 0)

    accelerator = Accelerator(nprocs=nprocs)
    dummy_dist_worker = accelerator.distribute(dummy_worker)

    dummy_dist_worker(run_processes)
    assert run_processes.value == nprocs


def test_prepare_batch_dict() -> None:
    accelerator = Accelerator(nprocs=1)
    accelerator.execution.device = "cpu"
    accelerator.execution.data_dtype = torch.float32
    batch: Dict[str, torch.Tensor] = {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
    prepared = accelerator.prepare_batch(batch)
    assert isinstance(prepared, dict)
    assert prepared["x"].device.type == "cpu"
    assert prepared["y"].device.type == "cpu"


def test_prepare_batch_list() -> None:
    accelerator = Accelerator(nprocs=1)
    accelerator.execution.device = "cpu"
    accelerator.execution.data_dtype = torch.float32
    batch: List[torch.Tensor] = [torch.tensor([1.0]), torch.tensor([2.0])]
    prepared = accelerator.prepare_batch(batch)
    assert isinstance(prepared, tuple)
    for tensor in prepared:
        assert tensor.device.type == "cpu"


def test_prepare_batch_tensor() -> None:
    accelerator = Accelerator(nprocs=1)
    accelerator.execution.device = "cpu"
    accelerator.execution.data_dtype = torch.float32
    tensor: torch.Tensor = torch.tensor([1.0, 2.0])
    prepared = accelerator.prepare_batch(tensor)
    assert prepared.device.type == "cpu"
    assert prepared.dtype == torch.float32


def test_all_reduce_dict_not_initialized() -> None:
    accelerator = Accelerator(nprocs=1)
    with patch.object(dist, "is_initialized", return_value=False):
        input_dict: Dict[str, torch.Tensor] = {"a": torch.tensor(2.0)}
        result = accelerator.all_reduce_dict(input_dict)
    assert result["a"].item() == 2.0


def test_all_reduce_dict_initialized() -> None:
    accelerator = Accelerator(nprocs=1)
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "all_reduce", lambda t, op: t.mul_(2)),
    ):
        input_dict: Dict[str, torch.Tensor] = {"a": torch.tensor(2.0)}
        result = accelerator.all_reduce_dict(input_dict)
    assert result["a"].item() == 2.0


def test_log_warnings(capsys: pytest.LogCaptureFixture) -> None:
    setup_logger()
    accelerator = Accelerator(nprocs=2)
    accelerator.execution_type = ExecutionType.TORCHRUN
    accelerator._log_warnings()
    captured = capsys.readouterr()
    assert "Process was launched using `torchrun`" in captured.err


def test_is_class_method() -> None:
    accelerator = Accelerator(nprocs=1)

    class Dummy:
        def method(self) -> None:
            pass

    dummy = Dummy()
    result = accelerator.is_class_method(Dummy.method, (dummy, 1))
    assert result is True

    def func() -> None:
        pass

    result = accelerator.is_class_method(func, ())
    assert result is False


def test_distribute_decorator() -> None:
    accelerator = Accelerator(nprocs=1)
    calls: List[str] = []

    def dummy_fun(*args: Any, **kwargs: Any) -> None:
        calls.append("called")

    decorated = accelerator.distribute(dummy_fun)
    decorated()
    assert calls == ["called"]


def test_distribute_decorator_with_class_method() -> None:
    accelerator = Accelerator(nprocs=1)

    class Dummy:
        def __init__(self) -> None:
            self.model: str = "model"
            self.optimizer: str = "optimizer"
            self.accelerator: Accelerator = accelerator

        @accelerator.distribute
        def train(self) -> str:
            return "trained"  # This return value will be ignored by the decorator

    dummy = Dummy()
    with patch.object(accelerator, "_spawn_method", lambda *args, **kwargs: None):
        result = dummy.train()
    assert result == ("model", "optimizer")
