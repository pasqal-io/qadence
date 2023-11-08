from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def numpy_to_tensor(
    x: np.ndarray,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> Tensor:
    """This only copies the numpy array if device or dtype are different than the ones of x."""
    return torch.as_tensor(x, dtype=dtype, device=device).requires_grad_(requires_grad)


def promote_to_tensor(x: Tensor | np.ndarray | float, requires_grad: bool = True) -> Tensor:
    """Convert the given type into a torch.Tensor."""
    if isinstance(x, float):
        return torch.tensor([[x]], requires_grad=requires_grad)
    elif isinstance(x, np.ndarray):
        return numpy_to_tensor(x, requires_grad=requires_grad)
    elif isinstance(x, Tensor):
        return x.requires_grad_(requires_grad)
    else:
        raise ValueError(f"Don't know how to promote {type(x)} to Tensor")


def promote_to(x: Tensor, dtype: Any) -> float | np.ndarray | Tensor:
    if dtype == float:
        assert x.size() == (1, 1)
        return x[0, 0].item()
    elif dtype == np.ndarray:
        return x.detach().cpu().numpy()
    elif dtype == Tensor:
        return x
    else:
        raise ValueError(f"Don't know how to convert Tensor to {dtype}")
