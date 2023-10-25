from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from torch import Tensor
from torch import device as torchdevice
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DictDataLoader:
    """This class only holds a dictionary of `DataLoader`s and samples from them"""

    dataloaders: dict[str, DataLoader]

    # this flag indicates that the dictionary contains dataloaders
    # which can automatically iterate at each epoch without having to
    # redefine the iterator itself (so basically no StopIteration exception
    # will occur). This is the case of the Flow library where the dataloader
    # is actually mostly used, so it is set to True by default
    has_automatic_iter: bool = True

    def __iter__(self) -> DictDataLoader:
        self.iters = {key: iter(dl) for key, dl in self.dataloaders.items()}
        return self

    def __next__(self) -> dict[str, Tensor]:
        return {key: next(it) for key, it in self.iters.items()}

    def to(self, device: torchdevice) -> DictDataLoader:
        self.iters = {
            key: DataLoader([dataloader_to_device(t, device=device) for t in dl])
            for key, dl in self.dataloaders.items()
        }
        return self


DataLoaderType = Union[DictDataLoader, DataLoader, list[Tensor], tuple[Tensor, Tensor], None]


def to_dataloader(x: Tensor, y: Tensor, batch_size: int = 1) -> DataLoader:
    """Convert two torch tensors x and y to a Dataloader."""
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def dataloader_to_device(
    dataloader: DataLoaderType, device: torchdevice = torchdevice("cpu")
) -> DataLoaderType:
    if isinstance(dataloader, Tensor):
        return dataloader.to(device=device)
    elif isinstance(dataloader, (list, tuple)):
        return list([t.to(device=device) for t in dataloader])
    elif isinstance(dataloader, (DataLoader)):
        return DataLoader([dataloader_to_device(t, device=device) for t in dataloader])
    elif isinstance(dataloader, DictDataLoader):
        return dataloader.to(device=device)
    else:
        return dataloader
