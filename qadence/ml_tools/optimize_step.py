from __future__ import annotations

from typing import Any, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer


def optimize_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    xs: dict | list | torch.Tensor | None,
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

    Returns:
        tuple: tuple containing the model, the optimizer, a dictionary with
            the collected metrics and the compute value loss
    """

    loss, metrics = None, {}

    def closure() -> Any:
        # NOTE: We need the nonlocal as we can't return a metric dict and
        # because e.g. LBFGS calls this closure multiple times but for some
        # reason the returned loss is always the first one...
        nonlocal metrics, loss
        optimizer.zero_grad()
        loss, metrics = loss_fn(model, xs)
        loss.backward(retain_graph=True)
        return loss.item()

    optimizer.step(closure)
    # return the loss/metrics that are being mutated inside the closure...
    return loss, metrics
