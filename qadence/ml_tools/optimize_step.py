from __future__ import annotations

from typing import Any, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer

from qadence.ml_tools.data import data_to_device


def optimize_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    xs: dict | list | torch.Tensor | None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor | float, dict | None]:
    """Default Torch optimize step with closure.

    This is the default optimization step which should work for most
    of the standard use cases of optimization of Torch models

    Args:
        model (Module): The input model
        optimizer (Optimizer): The chosen Torch optimizer
        loss_fn (Callable): A custom loss function
        xs (dict | list | torch.Tensor | None): the input data. If None it means
            that the given model does not require any input data
        device (torch.device): A target device to run computation on.
        dtype (torch.dtype): Data type for xs conversion.

    Returns:
        tuple: tuple containing the computed loss value, and a dictionary with
            the collected metrics.
    """

    loss, metrics = None, {}
    xs_to_device = data_to_device(xs, device=device, dtype=dtype)

    def closure() -> Any:
        # NOTE: We need the nonlocal as we can't return a metric dict and
        # because e.g. LBFGS calls this closure multiple times but for some
        # reason the returned loss is always the first one...
        nonlocal metrics, loss
        optimizer.zero_grad()
        loss, metrics = loss_fn(model, xs_to_device)
        loss.backward(retain_graph=True)
        return loss.item()

    optimizer.step(closure)
    # return the loss/metrics that are being mutated inside the closure...
    return loss, metrics
