import torch
import torch.nn as nn
from typing import Union, Callable
from typing import Tuple, Dict

def mse_loss(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    criterion = nn.MSELoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    metrics = {"mse": loss}
    return loss, metrics

def cross_entropy_loss(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    metrics = {"cross_entropy": loss}
    return loss, metrics


def get_loss_fn(loss_fn: Union[None, Callable, str]) -> Callable:
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
