from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatch
from itertools import cycle
from typing import Any, Iterator

from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch import Tensor
from torch import device as torch_device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset, TensorDataset


@dataclass
class OptimizeResult:
    """OptimizeResult stores many optimization intermediate values.

    We store at a current iteration,
    the model, optimizer, loss values, metrics. An extra dict
    can be used for saving other information to be used for callbacks.
    """

    iteration: int
    """Current iteration number."""
    model: Module
    """Model at iteration."""
    optimizer: Optimizer | NGOptimizer
    """Optimizer at iteration."""
    loss: Tensor | float | None = None
    """Loss value."""
    metrics: dict = field(default_factory=lambda: dict())
    """Metrics that can be saved during training."""
    extra: dict = field(default_factory=lambda: dict())
    """Extra dict for saving anything else to be used in callbacks."""


@dataclass
class DictDataLoader:
    """This class only holds a dictionary of `DataLoader`s and samples from them."""

    dataloaders: dict[str, DataLoader]

    def __iter__(self) -> DictDataLoader:
        self.iters = {key: iter(dl) for key, dl in self.dataloaders.items()}
        return self

    def __next__(self) -> dict[str, Tensor]:
        return {key: next(it) for key, it in self.iters.items()}


class InfiniteTensorDataset(IterableDataset):
    def __init__(self, *tensors: Tensor):
        """Randomly sample points from the first dimension of the given tensors.

        Behaves like a normal torch `Dataset` just that we can sample from it as
        many times as we want.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        from qadence.ml_tools.data import InfiniteTensorDataset

        x_data, y_data = torch.rand(5,2), torch.ones(5,1)
        # The dataset accepts any number of tensors with the same batch dimension
        ds = InfiniteTensorDataset(x_data, y_data)

        # call `next` to get one sample from each tensor:
        xs = next(iter(ds))
        print(str(xs)) # markdown-exec: hide
        ```
        """
        self.tensors = tensors

    def __iter__(self) -> Iterator:
        if len(set([t.size(0) for t in self.tensors])) != 1:
            raise ValueError("Size of first dimension must be the same for all tensors.")

        for idx in cycle(range(self.tensors[0].size(0))):
            yield tuple(t[idx] for t in self.tensors)


def to_dataloader(*tensors: Tensor, batch_size: int = 1, infinite: bool = False) -> DataLoader:
    """Convert torch tensors an (infinite) Dataloader.

    Arguments:
        *tensors: Torch tensors to use in the dataloader.
        batch_size: batch size of sampled tensors
        infinite: if `True`, the dataloader will keep sampling indefinitely even after the whole
            dataset was sampled once

    Examples:

    ```python exec="on" source="above" result="json"
    import torch
    from qadence.ml_tools import to_dataloader

    (x, y, z) = [torch.rand(10) for _ in range(3)]
    loader = iter(to_dataloader(x, y, z, batch_size=5, infinite=True))
    print(next(loader))
    print(next(loader))
    print(next(loader))
    ```
    """
    ds = InfiniteTensorDataset(*tensors) if infinite else TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size)


@singledispatch
def data_to_device(xs: Any, *args: Any, **kwargs: Any) -> Any:
    """Utility method to move arbitrary data to 'device'."""
    raise ValueError(f"Unable to move {type(xs)} with input args: {args} and kwargs: {kwargs}.")


@data_to_device.register
def _(xs: None, *args: Any, **kwargs: Any) -> None:
    return xs


@data_to_device.register(Tensor)
def _(xs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return xs.to(*args, **kwargs)


@data_to_device.register(list)
def _(xs: list, *args: Any, **kwargs: Any) -> list:
    return [data_to_device(x, *args, **kwargs) for x in xs]


@data_to_device.register(dict)
def _(xs: dict, *args: Any, **kwargs: Any) -> dict:
    return {key: data_to_device(val, *args, **kwargs) for key, val in xs.items()}


@data_to_device.register(DataLoader)
def _(xs: DataLoader, *args: Any, **kwargs: Any) -> DataLoader:
    return DataLoader(data_to_device(xs.dataset, *args, **kwargs))


@data_to_device.register(DictDataLoader)
def _(xs: DictDataLoader, device: torch_device) -> DictDataLoader:
    return DictDataLoader({key: data_to_device(val, device) for key, val in xs.dataloaders.items()})


@data_to_device.register(InfiniteTensorDataset)
def _(xs: InfiniteTensorDataset, device: torch_device) -> InfiniteTensorDataset:
    return InfiniteTensorDataset(*[data_to_device(val, device) for val in xs.tensors])


@data_to_device.register(TensorDataset)
def _(xs: TensorDataset, device: torch_device) -> TensorDataset:
    return TensorDataset(*[data_to_device(val, device) for val in xs.tensors])
