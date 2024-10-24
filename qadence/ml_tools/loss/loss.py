import torch
import torch.nn as nn
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

