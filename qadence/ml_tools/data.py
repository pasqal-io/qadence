from __future__ import annotations

from dataclasses import dataclass

import torch
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

    def __next__(self) -> dict[str, torch.Tensor]:
        return {key: next(it) for key, it in self.iters.items()}


def to_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 1) -> DataLoader:
    """Convert two torch tensors x and y to a Dataloader."""
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)
