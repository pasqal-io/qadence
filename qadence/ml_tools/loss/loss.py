from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def mse_loss(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Mean Squared Error (MSE) loss between model predictions and targets.

    Args:
        model (nn.Module): The PyTorch model used for generating predictions.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - inputs (torch.Tensor): The input data.
            - targets (torch.Tensor): The ground truth labels.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed MSE loss value.
            - metrics (dict[str, float]): A dictionary with the MSE loss value.
    """
    criterion = nn.MSELoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    metrics = {"mse": loss}
    return loss, metrics


def cross_entropy_loss(
    model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the Cross Entropy loss between model predictions and targets.

    Args:
        model (nn.Module): The PyTorch model used for generating predictions.
        batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
            - inputs (torch.Tensor): The input data.
            - targets (torch.Tensor): The ground truth labels.

    Returns:
        Tuple[torch.Tensor, dict[str, float]]:
            - loss (torch.Tensor): The computed Cross Entropy loss value.
            - metrics (dict[str, float]): A dictionary with the Cross Entropy loss value.
    """
    criterion = nn.CrossEntropyLoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    metrics = {"cross_entropy": loss}
    return loss, metrics


def get_loss_fn(loss_fn: str | Callable | None) -> Callable:
    """
    Returns the appropriate loss function based on the input argument.

    Args:
        loss_fn (str | Callable | None): The loss function to use.
            - If `loss_fn` is a callable, it will be returned directly.
            - If `loss_fn` is a string, it should be one of:
                - "mse": Returns the `mse_loss` function.
                - "cross_entropy": Returns the `cross_entropy_loss` function.
            - If `loss_fn` is `None`, the default `mse_loss` function will be returned.

    Returns:
        Callable: The corresponding loss function.

    Raises:
        ValueError: If `loss_fn` is a string but not a supported loss function name.
    """
    if callable(loss_fn):
        return loss_fn
    elif isinstance(loss_fn, str):
        if loss_fn == "mse":
            return mse_loss
        elif loss_fn == "cross_entropy":
            return cross_entropy_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")
    else:
        return mse_loss
