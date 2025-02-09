from __future__ import annotations

from typing import Any, Callable

import nevergrad as ng
import torch
from torch.nn import Module
from torch.optim import Optimizer

from qadence.ml_tools.data import data_to_device
from qadence.ml_tools.parameters import set_parameters
from qadence.ml_tools.tensors import promote_to_tensor


def optimize_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    xs: dict | list | torch.Tensor | None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor | float, dict | None]:
    """Default Torch optimize step with closure.

    This is the default optimization step.

    Args:
        model (Module): The input model to be optimized.
        optimizer (Optimizer): The chosen Torch optimizer.
        loss_fn (Callable): A custom loss function
            that returns the loss value and a dictionary of metrics.
        xs (dict | list | Tensor | None): The input data. If None, it means
            the given model does not require any input data.
        device (torch.device): A target device to run computations on.
        dtype (torch.dtype): Data type for `xs` conversion.

    Returns:
        tuple[Tensor | float, dict | None]: A tuple containing the computed loss value
            and a dictionary with collected metrics.
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


def update_ng_parameters(
    model: Module,
    optimizer: ng.optimizers.Optimizer,
    loss_fn: Callable[[Module, torch.Tensor | None], tuple[float, dict]],
    data: torch.Tensor | None,
    ng_params: ng.p.Array,
) -> tuple[float, dict, ng.p.Array]:
    """Update the model parameters using Nevergrad.

    This function integrates Nevergrad for derivative-free optimization.

    Args:
        model (Module): The PyTorch model to be optimized.
        optimizer (ng.optimizers.Optimizer): A Nevergrad optimizer instance.
        loss_fn (Callable[[Module, Tensor | None], tuple[float, dict]]): A custom loss function
            that returns the loss value and a dictionary of metrics.
        data (Tensor | None): Input data for the model. If None, it means the model does
            not require input data.
        ng_params (ng.p.Array): The current set of parameters managed by Nevergrad.

    Returns:
        tuple[float, dict, ng.p.Array]: A tuple containing the computed loss value,
            a dictionary of metrics, and the updated Nevergrad parameters.
    """
    loss, metrics = loss_fn(model, data)  # type: ignore[misc]
    optimizer.tell(ng_params, float(loss))
    ng_params = optimizer.ask()  # type: ignore[assignment]
    params = promote_to_tensor(ng_params.value, requires_grad=False)
    set_parameters(model, params)
    return loss, metrics, ng_params
