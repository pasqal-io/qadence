from __future__ import annotations

from .callbacks.saveload import load_checkpoint, load_model, write_checkpoint
from .config import AnsatzConfig, FeatureMapConfig, TrainConfig
from .constructors import create_ansatz, create_fm_blocks, observable_from_config
from .data import DictDataLoader, InfiniteTensorDataset, OptimizeResult, to_dataloader
from .models import QNN
from .optimize_step import optimize_step as default_optimize_step
from .parameters import get_parameters, num_parameters, set_parameters
from .tensors import numpy_to_tensor, promote_to, promote_to_tensor
from .trainer import Trainer

# Modules to be automatically added to the qadence namespace
__all__ = [
    "AnsatzConfig",
    "create_ansatz",
    "create_fm_blocks",
    "DictDataLoader",
    "FeatureMapConfig",
    "load_checkpoint",
    "observable_from_config",
    "QNN",
    "TrainConfig",
    "OptimizeResult",
    "Trainer",
    "write_checkpoint",
]
