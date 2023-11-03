from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from functools import singledispatch
from typing import Any, Union

from torch import Tensor, is_tensor
from torch import device as torchdevice
from torch.utils.data import DataLoader, TensorDataset, IterableDataset


@dataclass
class DictDataLoader:
    """This class only holds a dictionary of `DataLoader`s and samples from them"""

    dataloaders: dict[str, DataLoader]

    def __iter__(self) -> DictDataLoader:
        self.iters = {key: iter(dl) for key, dl in self.dataloaders.items()}
        return self

    def __next__(self) -> dict[str, Tensor]:
        return {key: next(it) for key, it in self.iters.items()}


class InfiniteTensorDataset(IterableDataset):
    def __init__(self, *tensors: torch.Tensor):
        """Randomly sample points from the first dimension of the given tensors.
        Behaves like a normal torch `Dataset` just that we can sample from it as
        many times as we want.

        Examples:
        ```python exec="on" source="above" result="json"
        import torch
        from flow.data import InfiniteTensorDataset

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


def to_dataloader(x: Tensor, y: Tensor, batch_size: int = 1, infinite=False) -> DataLoader:
    """Convert two torch tensors x and y to a Dataloader."""
    ds = InfiniteTensorDataset(x, y) if infinite else TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


@singledispatch
def data_to_device(xs: Any, device: str) -> Any:
    raise ValueError(f"Cannot move {type(xs)} to a pytorch device.")


@data_to_device.register
def _(xs: Tensor, device: str) -> Tensor:
    return xs.to(device, non_blocking=True)


@data_to_device.register
def _(xs: list, device: str) -> list:
    return [data_to_device(x, device) for x in xs]


@data_to_device.register
def _(xs: dict, device: str) -> dict:
    return {key: data_to_device(val, device) for key, val in xs.items()}
