from __future__ import annotations

from .loss import cross_entropy_loss, get_loss_fn, mse_loss

# Modules to be automatically added to the qadence.ml_tools.loss namespace
__all__ = [
    "cross_entropy_loss",
    "get_loss_fn",
    "mse_loss",
]
