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
from typing import Any, Callable, Dict, Generator, Iterator, List, Tuple

from qadence.ml_tools.train_utils.accelerator import Accelerator
from qadence.ml_tools.data import DictDataLoader, to_dataloader


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


@pytest.fixture
def mock_mp_spawn() -> Iterator[None]:
    def fake_mp_spawn(
        fn: Callable[[int, Any, Callable[..., None], Any, Dict[Any, Any]], None],
        spawn_args: Any,
        nprocs: int,
        join: bool,
    ) -> None:
        fn(0, None, lambda *a, **kw: None, spawn_args, {})

    with patch.object(mp, "spawn", fake_mp_spawn):
        yield


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ml_tools")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


def test_prepare_model_without_ddp() -> None:
    accelerator = Accelerator(spawn=False, nprocs=1, compute_setup="cpu")
    accelerator.world_size = 1
    accelerator.device = "cpu"
    model = DummyModel()
    prepared_model = accelerator._prepare_model(model)
    assert not isinstance(prepared_model, DDP)
    for param in prepared_model.parameters():
        assert param.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prepare_model_with_ddp_cuda() -> None:
    accelerator = Accelerator(spawn=False, nprocs=2, compute_setup="gpu")
    accelerator.world_size = 2
    accelerator.local_rank = 0
    accelerator.device = "cuda:0"
    model = DummyModel()
    prepared_model = accelerator._prepare_model(model)
    assert isinstance(prepared_model, DDP)
    for param in prepared_model.parameters():
        assert param.device.type == "cuda"


def test_prepare_optimizer() -> None:
    accelerator = Accelerator(spawn=False)
    model = DummyModel()
    optimizer = get_dummy_optimizer(model)
    prepared_optimizer = accelerator._prepare_optimizer(optimizer)
    assert prepared_optimizer is optimizer


def test_prepare_data_without_distributed() -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.world_size = 1
    dl = get_dummy_dataloader()
    prepared_dl = accelerator._prepare_data(dl)
    assert prepared_dl is dl


def test_prepare_data_with_distributed() -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.world_size = 2
    accelerator.local_rank = 0
    dl = get_dummy_dataloader()
    prepared_dl = accelerator._prepare_data(dl)
    assert isinstance(prepared_dl.sampler, DistributedSampler)  # type: ignore[union-attr]


def test_prepare_dict_dataloader(dict_dataloader: DictDataLoader) -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.world_size = 2
    accelerator.local_rank = 0
    prepared_dl = accelerator._prepare_data(dict_dataloader)
    assert isinstance(prepared_dl, DictDataLoader)
    for name, dl in prepared_dl.dataloaders.items():
        assert isinstance(dl.sampler, DistributedSampler)


def test_prepare_function(
    dummy_model: DummyModel, dummy_optimizer: optim.Optimizer, dummy_dataloader: DataLoader
) -> None:
    accelerator = Accelerator(spawn=False)
    model_prepped, opt_prepped, dl_prepped = accelerator.prepare(
        dummy_model, dummy_optimizer, dummy_dataloader
    )
    assert isinstance(model_prepped, nn.Module)
    assert isinstance(opt_prepped, optim.Optimizer)
    assert isinstance(dl_prepped, DataLoader)


def test_prepare_function_with_spawn(
    mock_mp_spawn: Any,
    dummy_model: DummyModel,
    dummy_optimizer: optim.Optimizer,
    dummy_dataloader: DataLoader,
) -> None:
    accelerator = Accelerator(spawn=True, nprocs=2)
    model_prepped, opt_prepped, dl_prepped = accelerator.prepare(
        dummy_model, dummy_optimizer, dummy_dataloader
    )
    assert isinstance(model_prepped, nn.Module)
    assert isinstance(opt_prepped, optim.Optimizer)
    assert isinstance(dl_prepped, DataLoader)


def test_prepare_batch_dict() -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.device = "cpu"
    accelerator.data_dtype = torch.float32
    batch: Dict[str, torch.Tensor] = {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
    prepared = accelerator.prepare_batch(batch)
    assert isinstance(prepared, dict)
    assert prepared["x"].device.type == "cpu"
    assert prepared["y"].device.type == "cpu"


def test_prepare_batch_list() -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.device = "cpu"
    accelerator.data_dtype = torch.float32
    batch: List[torch.Tensor] = [torch.tensor([1.0]), torch.tensor([2.0])]
    prepared = accelerator.prepare_batch(batch)
    assert isinstance(prepared, tuple)
    for tensor in prepared:
        assert tensor.device.type == "cpu"


def test_prepare_batch_tensor() -> None:
    accelerator = Accelerator(spawn=False)
    accelerator.device = "cpu"
    accelerator.data_dtype = torch.float32
    tensor: torch.Tensor = torch.tensor([1.0, 2.0])
    prepared = accelerator.prepare_batch(tensor)
    assert prepared.device.type == "cpu"
    assert prepared.dtype == torch.float32


def test_all_reduce_dict_not_initialized() -> None:
    accelerator = Accelerator(spawn=False)
    with patch.object(dist, "is_initialized", return_value=False):
        input_dict: Dict[str, torch.Tensor] = {"a": torch.tensor(2.0)}
        result = accelerator.all_reduce_dict(input_dict)
    assert result["a"].item() == 2.0


def test_all_reduce_dict_initialized() -> None:
    accelerator = Accelerator(spawn=False)
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
    accelerator = Accelerator(spawn=True)
    accelerator.strategy = "torchrun"
    accelerator._log_warnings()
    captured = capsys.readouterr()
    assert "Spawn mode is enabled" in captured.err
    assert not accelerator.spawn


def test_is_class_method() -> None:
    accelerator = Accelerator(spawn=False)

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
    accelerator = Accelerator(spawn=False)
    calls: List[str] = []

    def dummy_fun(*args: Any, **kwargs: Any) -> None:
        calls.append("called")

    decorated = accelerator.distribute(dummy_fun)
    decorated()
    assert calls == ["called"]


def test_distribute_decorator_with_class_method() -> None:
    accelerator = Accelerator(spawn=False)

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
