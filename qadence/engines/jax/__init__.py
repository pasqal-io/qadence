from __future__ import annotations

from jax import config

from .differentiable_backend import DifferentiableBackend
from .differentiable_expectation import DifferentiableExpectation

config.update("jax_enable_x64", True)
