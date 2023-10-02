from __future__ import annotations

from functools import singledispatch
from typing import Any, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer


@singledispatch
def data_to_model(xs: Any, device: str = "cpu") -> Any:
    """Default behavior for single-dispatched function

    Just return the given data independently on the type

    Args:
        xs (Any): the input data
        device (str, optional): The torch device. Not used in this implementation.

    Returns:
        Any: the `xs` argument untouched
    """
    return xs


@data_to_model.register(list)
def _(xs: list, device: str = "cpu") -> list:
    xs_to_device = xs

    for x in xs_to_device:
        if torch.is_tensor(x):
            x.to(device, non_blocking=True)

    return xs_to_device


@data_to_model.register(dict)
def _(xs: dict, device: str = "cpu") -> dict:
    # TODO: Make sure that they are tensors before calling .to() method
    to_device = {key: [x.to(device, non_blocking=True) for x in val] for key, val in xs.items()}
    return to_device


def optimize_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    xs: dict | list | torch.Tensor | None,
    device: str = "cpu",
) -> tuple[torch.Tensor | float, dict | None]:
    """Default Torch optimize step with closure

    This is the default optimization step which should work for most
    of the standard use cases of optimization of Torch models

    Args:
        model (Module): The input model
        optimizer (Optimizer): The chosen Torch optimizer
        loss_fn (Callable): A custom loss function
        xs (dict | list | torch.Tensor | None): the input data. If None it means
            that the given model does not require any input data
        device (str, optional): The device were computations are executed.
            Defaults to "cpu".

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
