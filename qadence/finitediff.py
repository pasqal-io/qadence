from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def finitediff(
    f: Callable,
    x: Tensor,
    derivative_indices: tuple[int, ...],
    eps: float = None,
) -> Tensor:
    """
    Arguments:

        f: Function to differentiate
        x: Input of shape `(batch_size, input_size)`
        derivative_indices: which *input* to differentiate (i.e. which variable x[:,i])
        eps: finite difference spacing (uses `torch.finfo(x.dtype).eps ** (1 / (2 + order))` as a
            default)
    """

    if eps is None:
        order = len(derivative_indices)
        eps = torch.finfo(x.dtype).eps ** (1 / (2 + order))

    # compute derivative direction vector(s)
    eps = torch.as_tensor(eps, dtype=x.dtype)
    _eps = 1 / eps  # type: ignore[operator]
    ev = torch.zeros_like(x)
    i = derivative_indices[0]
    ev[:, i] += eps

    # recursive finite differencing for higher order than 3 / mixed derivatives
    if len(derivative_indices) > 3 or len(set(derivative_indices)) > 1:
        di = derivative_indices[1:]
        return (finitediff(f, x + ev, di) - finitediff(f, x - ev, di)) * _eps / 2
    elif len(derivative_indices) == 3:
        return (f(x + 2 * ev) - 2 * f(x + ev) + 2 * f(x - ev) - f(x - 2 * ev)) * _eps**3 / 2
    elif len(derivative_indices) == 2:
        return (f(x + ev) + f(x - ev) - 2 * f(x)) * _eps**2
    elif len(derivative_indices) == 1:
        return (f(x + ev) - f(x - ev)) * _eps / 2
    else:
        raise ValueError("If you see this error there is a bug in the `finitediff` function.")
