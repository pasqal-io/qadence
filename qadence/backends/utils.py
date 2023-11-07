from __future__ import annotations

from collections import Counter
from math import log2, prod
from typing import Callable, Sequence

import numpy as np
import torch
from pyqtorch.apply import apply_operator
from pyqtorch.parametric import Parametric as PyQParametric
from pyqtorch.utils import overlap
from torch import Tensor

from qadence.utils import Endianness, int_to_basis

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
    """Convert the given type inco a torch.Tensor"""
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


def to_list_of_dicts(param_values: dict[str, Tensor]) -> list[dict[str, float]]:
    if not param_values:
        return [param_values]

    max_batch_size = max(p.size()[0] for p in param_values.values())
    batched_values = {
        k: (v if v.size()[0] == max_batch_size else v.repeat(max_batch_size, 1))
        for k, v in param_values.items()
    }

    return [{k: v[i] for k, v in batched_values.items()} for i in range(max_batch_size)]


def finitediff_sampling(
    f: Callable, x: torch.Tensor, eps: float = FINITE_DIFF_EPS, num_samples: int = 10
) -> torch.Tensor:
    def _finitediff(val: torch.Tensor) -> torch.Tensor:
        return (f(x + val) - f(x - val)) / (2 * val)  # type: ignore

    with torch.no_grad():
        return torch.mean(
            torch.cat([_finitediff(val) for val in torch.rand(1) for _ in range(num_samples)])
        )


def finitediff(f: Callable, x: torch.Tensor, eps: float = FINITE_DIFF_EPS) -> torch.Tensor:
    return (f(x + eps) - f(x - eps)) / (2 * eps)  # type: ignore


def dydx(
    jacobian: torch.Tensor,
    qubit_support: tuple,
    out_state: torch.Tensor,
    projected_state: torch.Tensor,
) -> torch.Tensor:
    return 2 * overlap(
        projected_state,
        apply_operator(
            state=out_state,
            operator=jacobian,
            qubits=qubit_support,
        ),
    )


def dydxx(
    op: PyQParametric,
    values: dict[str, torch.Tensor],
    out_state: torch.Tensor,
    projected_state: torch.Tensor,
) -> torch.Tensor:
    return 2 * finitediff_sampling(
        lambda val: dydx(
            op.jacobian({op.param_name: val}), op.qubit_support, out_state, projected_state
        ),
        values[op.param_name],
    )


def pyqify(state: torch.Tensor, n_qubits: int = None) -> torch.Tensor:
    if n_qubits is None:
        n_qubits = int(log2(state.shape[1]))
    if (state.ndim != 2) or (state.size(1) != 2**n_qubits):
        raise ValueError(
            "The initial state must be composed of tensors of size "
            f"(batch_size, 2**n_qubits). Found: {state.size() = }."
        )
    return state.T.reshape([2] * n_qubits + [state.size(0)])


def unpyqify(state: torch.Tensor) -> torch.Tensor:
    return torch.flatten(state, start_dim=0, end_dim=-2).t()


def validate_pyq_state(state: torch.Tensor, n_qubits: int) -> torch.Tensor:
    if prod(state.size()[:-1]) != 2**n_qubits:
        raise ValueError(
            "A pyqified initial state must be composed of tensors of size "
            f"(2, 2, ..., batch_size). Found: {state.size() = }."
        )
    else:
        return state


def infer_batchsize(param_values: dict[str, torch.Tensor] = None) -> int:
    return max([len(tensor) for tensor in param_values.values()]) if param_values else 1
