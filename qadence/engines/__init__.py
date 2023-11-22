# flake8: noqa F401
from __future__ import annotations

from .differentiable_backend import DifferentiableBackend
from .jax.differentiable_backend import JaxBackend
from .torch.differentiable_backend import TorchBackend

# Modules to be automatically added to the qadence namespace
__all__ = ["DifferentiableBackend"]
