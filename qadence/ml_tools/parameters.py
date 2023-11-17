from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module


def get_parameters(model: Module) -> Tensor:
    """Retrieve all trainable model parameters in a single vector.

    Args:
        model (Module): the input PyTorch model

    Returns:
        Tensor: a 1-dimensional tensor with the parameters
    """
    ps = [p.reshape(-1) for p in model.parameters() if p.requires_grad]
    return torch.concat(ps)


def set_parameters(model: Module, theta: Tensor) -> None:
    """Set all trainable parameters of a model from a single vector.

    Notice that this function assumes prior knowledge of right number
    of parameters in the model

    Args:
        model (Module): the input PyTorch model
        theta (Tensor): the parameters to assign
    """

    with torch.no_grad():
        idx = 0
        for ps in model.parameters():
            if ps.requires_grad:
                n = torch.numel(ps)
                if ps.ndim == 0:
                    ps[()] = theta[idx : idx + n]
                else:
                    ps[:] = theta[idx : idx + n].reshape(ps.size())
                idx += n


def num_parameters(model: Module) -> int:
    """Return the total number of parameters of the given model."""
    return len(get_parameters(model))
