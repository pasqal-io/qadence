from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array, device_get
from sympy import Expr
from sympy2jax import SymbolicModule as JaxSympyModule
from torch import Tensor, cdouble, from_numpy


def jarr_to_tensor(arr: Array, dtype: Any = cdouble) -> Tensor:
    return from_numpy(device_get(arr)).to(dtype=dtype)


def tensor_to_jnp(tensor: Tensor, dtype: Any = jnp.complex128) -> Array:
    return (
        jnp.array(tensor.numpy(), dtype=dtype)
        if not tensor.requires_grad
        else jnp.array(tensor.detach().numpy(), dtype=dtype)
    )


def values_to_jax(param_values: dict[str, Tensor]) -> dict[str, Array]:
    return {key: jnp.array(value.detach().numpy()) for key, value in param_values.items()}


def jaxify(expr: Expr) -> JaxSympyModule:
    return JaxSympyModule(expr)


def unhorqify(state: Array) -> Array:
    """Convert a state of shape [2] * n_qubits + [batch_size] to (batch_size, 2**n_qubits)."""
    return jnp.ravel(state, start_dim=0, end_dim=-2).t()
