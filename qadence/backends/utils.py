from __future__ import annotations

from collections import Counter
from math import log2
from typing import Callable, Sequence

import numpy as np
import pyqtorch as pyq
import torch
from numpy.typing import ArrayLike
from pyqtorch.apply import apply_operator
from pyqtorch.primitives import Parametric as PyQParametric
from pyqtorch.utils import DensityMatrix
from torch import (
    Tensor,
    cat,
    complex64,
    complex128,
    mean,
    no_grad,
    rand,
)

from qadence.types import Endianness, ParamDictType
from qadence.utils import int_to_basis, is_qadence_shape

FINITE_DIFF_EPS = 1e-06
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    int: torch.int64,
    float: torch.float64,
    complex: torch.complex128,
}


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


def numpy_to_tensor(
    x: np.ndarray,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.complex128,
    requires_grad: bool = False,
) -> Tensor:
    """This only copies the numpy array if device or dtype are different than the ones of x."""
    return torch.as_tensor(x, dtype=dtype, device=device).requires_grad_(requires_grad)


def promote_to_tensor(
    x: Tensor | np.ndarray | float,
    dtype: torch.dtype = torch.complex128,
    requires_grad: bool = True,
) -> Tensor:
    """Convert the given type into a torch.Tensor."""
    if isinstance(x, float):
        return torch.tensor([[x]], dtype=dtype, requires_grad=requires_grad)
    elif isinstance(x, np.ndarray):
        return numpy_to_tensor(
            x, dtype=numpy_to_torch_dtype_dict.get(x.dtype), requires_grad=requires_grad
        )
    elif isinstance(x, Tensor):
        return x.requires_grad_(requires_grad)
    else:
        raise ValueError(f"Don't know how to promote {type(x)} to Tensor")


# FIXME: Not being used, maybe remove in v1.0.0
def count_bitstrings(sample: Tensor, endianness: Endianness = Endianness.BIG) -> Counter:
    # Convert to a tensor of integers.
    n_qubits = sample.size()[1]
    base = torch.ones(n_qubits, dtype=torch.int64) * 2
    powers_of_2 = torch.pow(base, reversed(torch.arange(n_qubits)))
    int_tensor = torch.matmul(sample, powers_of_2)
    # Count occurences of integers.
    count_int = torch.bincount(int_tensor)
    # Return a Counter for non-empty bitstring counts.
    return Counter(
        {
            int_to_basis(k=k, n_qubits=n_qubits, endianness=endianness): count.item()
            for k, count in enumerate(count_int)
            if count > 0
        }
    )


def to_list_of_dicts(param_values: ParamDictType) -> list[ParamDictType]:
    if not param_values:
        return [param_values]

    max_batch_size = max(p.size()[0] for p in param_values.values() if isinstance(p, Tensor))
    batched_values = {
        k: (v if v.size()[0] == max_batch_size else v.repeat(max_batch_size, 1))
        for k, v in param_values.items()
        if isinstance(v, Tensor)
    }

    return [{k: v[i] for k, v in batched_values.items()} for i in range(max_batch_size)]


def pyqify(state: Tensor, n_qubits: int = None) -> ArrayLike:
    """Convert a state of shape (batch_size, 2**n_qubits) to [2] * n_qubits + [batch_size]."""
    if n_qubits is None:
        n_qubits = int(log2(state.shape[1]))
    if len(state.shape) != 2 or (state.shape[1] != 2**n_qubits):
        raise ValueError(
            "The initial state must be composed of tensors/arrays of size "
            f"(batch_size, 2**n_qubits). Found: {state.shape = }."
        )
    return state.T.reshape([2] * n_qubits + [state.shape[0]])


def unpyqify(state: Tensor) -> Tensor:
    """
    Convert a state of shape [2] * n_qubits + [batch_size] to (batch_size, 2**n_qubits).

    Convert a density matrix of shape (2**n_qubits, 2**n_qubits, batch_size)
    to (batch_size, 2**n_qubits, 2**n_qubits)
    """
    if isinstance(state, DensityMatrix):
        return torch.einsum("ijk->kij", state)
    return torch.flatten(state, start_dim=0, end_dim=-2).t()


def is_pyq_shape(state: Tensor, n_qubits: int) -> bool:
    return state.size()[:-1] == [2] * n_qubits  # type: ignore[no-any-return]


def validate_state(state: Tensor, n_qubits: int) -> None:
    """Check if a custom initial state conforms to the qadence or the pyqtorch format."""
    if state.dtype not in [complex128, complex64]:
        raise TypeError(f"Expected complex dtype, got {state.dtype}")
    elif len(state.size()) < 2:
        raise ValueError(f"Invalid state shape. Got {state.shape}")
    elif not is_qadence_shape(state, n_qubits) and not is_pyq_shape(state, n_qubits):
        raise ValueError(
            f"Allowed formats for custom initial state are:\
                  (1) Qadence shape: (batch_size, 2**n_qubits)\
                  (2) Pyqtorch shape: (2 * n_qubits + [batch_size])\
                  Found: {state.size() = }"
        )


def infer_batchsize(param_values: dict[str, Tensor] = None) -> int:
    """Infer the batch_size through the length of the parameter tensors."""
    try:
        return (
            max(
                [
                    len(tensor_or_dict)
                    for tensor_or_dict in param_values.values()
                    if isinstance(tensor_or_dict, Tensor)
                ]
            )
            if param_values
            else 1
        )
    except Exception:
        return 1


# The following functions can be used to compute potentially higher order gradients using pyqtorch's
# native 'jacobian' methods.


def finitediff(
    f: Callable,
    x: Tensor,
    derivative_indices: tuple[int, ...],
    eps: float = None,
) -> Tensor:
    """
    Compute the finite difference of a function at a point.

    Args:
        f: The function to differentiate.
        x: Input of size `(batch_size, input_size)`.
        derivative_indices: Which *input* to differentiate (i.e. which variable x[:,i])
        eps: finite difference spacing (uses `torch.finfo(x.dtype).eps ** (1 / (2 + order))`
            as default)

    Returns:
        (Tensor): The finite difference of the function at the point `x`.
    """

    if eps is None:
        order = len(derivative_indices)
        eps = torch.finfo(x.dtype).eps ** (1 / (2 + order))

    # compute derivative direction vector(s)
    eps = torch.as_tensor(eps, dtype=x.dtype)
    _eps = 1 / eps  # type: ignore[operator]
    ev = torch.zeros_like(x)
    i = derivative_indices[0]
    ev[:, i] += eps

    # recursive finite differencing for higher order than 3 / mixed derivatives
    if len(derivative_indices) > 3 or len(set(derivative_indices)) > 1:
        di = derivative_indices[1:]
        return (finitediff(f, x + ev, di) - finitediff(f, x - ev, di)) * _eps / 2
    elif len(derivative_indices) == 3:
        return (f(x + 2 * ev) - 2 * f(x + ev) + 2 * f(x - ev) - f(x - 2 * ev)) * _eps**3 / 2
    elif len(derivative_indices) == 2:
        return (f(x + ev) + f(x - ev) - 2 * f(x)) * _eps**2
    elif len(derivative_indices) == 1:
        return (f(x + ev) - f(x - ev)) * _eps / 2
    else:
        raise ValueError("If you see this error there is a bug in the `finitediff` function.")


def finitediff_sampling(
    f: Callable, x: Tensor, eps: float = FINITE_DIFF_EPS, num_samples: int = 10
) -> Tensor:
    def _finitediff(val: Tensor) -> Tensor:
        return (f(x + val) - f(x - val)) / (2 * val)  # type: ignore

    with no_grad():
        return mean(cat([_finitediff(val) for val in rand(1) for _ in range(num_samples)]))


def dydx(
    jacobian: Tensor,
    qubit_support: tuple,
    out_state: Tensor,
    projected_state: Tensor,
) -> Tensor:
    return 2 * pyq.inner_prod(
        projected_state,
        apply_operator(
            state=out_state,
            operator=jacobian,
            qubits=qubit_support,
        ).real,
    )


def dydxx(
    op: PyQParametric,
    values: dict[str, Tensor],
    out_state: Tensor,
    projected_state: Tensor,
) -> Tensor:
    return 2 * finitediff_sampling(
        lambda val: dydx(
            op.jacobian({op.param_name: val}), op.qubit_support, out_state, projected_state
        ),
        values[op.param_name],
    )
