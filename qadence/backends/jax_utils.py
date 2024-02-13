from __future__ import annotations

from math import log2
from typing import Any

import jax.numpy as jnp
from jax import Array, device_get
from sympy import Expr
from sympy2jax import SymbolicModule as JaxSympyModule
from torch import Tensor, cdouble, from_numpy

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    KronBlock,
    ProjectorBlock,
    ScaleBlock,
)
from qadence.blocks.block_to_tensor import _gate_parameters
from qadence.types import Endianness, ParamDictType


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
    return jnp.ravel(state)


def horqify(state: Array) -> Array:
    n_qubits = int(log2(state.shape[1]))
    return state.reshape([2] * n_qubits)


def uniform_batchsize(param_values: ParamDictType) -> ParamDictType:
    max_batch_size = max(p.size for p in param_values.values())
    batched_values = {
        k: (v if v.size == max_batch_size else v.repeat(max_batch_size))
        for k, v in param_values.items()
    }
    return batched_values


IMAT = jnp.eye(2, dtype=jnp.cdouble)
ZEROMAT = jnp.zeros_like(IMAT)
XMAT = jnp.array([[0, 1], [1, 0]], dtype=jnp.cdouble)
YMAT = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.cdouble)
ZMAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.cdouble)

MAT_DICT = {"I": IMAT, "Z": ZMAT, "Y": YMAT, "X": XMAT}


def _fill_identities(
    block_mat: Array,
    qubit_support: tuple,
    full_qubit_support: tuple | list,
    diag_only: bool = False,
    endianness: Endianness = Endianness.BIG,
) -> Array:
    qubit_support = tuple(sorted(qubit_support))
    mat = IMAT if qubit_support[0] != full_qubit_support[0] else block_mat
    if diag_only:
        mat = jnp.diag(mat.squeeze(0))
    for i in full_qubit_support[1:]:
        if i == qubit_support[0]:
            other = jnp.diag(block_mat) if diag_only else block_mat
            mat = jnp.kron(mat, other)
        elif i not in qubit_support:
            other = jnp.diag(IMAT) if diag_only else IMAT
            mat = jnp.kron(mat, other)
    return mat


def block_to_jax(
    block: AbstractBlock,
    values: dict = None,
    qubit_support: tuple | None = None,
    use_full_support: bool = True,
    endianness: Endianness = Endianness.BIG,
) -> Array:
    if values is None:
        from qadence.blocks import embedding

        (ps, embed) = embedding(block)
        values = embed(ps, {})

    # get number of qubits
    if qubit_support is None:
        if use_full_support:
            qubit_support = tuple(range(0, block.n_qubits))
        else:
            qubit_support = block.qubit_support
    nqubits = len(qubit_support)

    if isinstance(block, (ChainBlock, KronBlock)):
        # create identity matrix of appropriate dimensions
        mat = IMAT
        for i in range(nqubits - 1):
            mat = jnp.kron(mat, IMAT)

        # perform matrix multiplications
        for b in block.blocks:
            other = block_to_jax(b, values, qubit_support, endianness=endianness)
            mat = jnp.matmul(other, mat)

    elif isinstance(block, AddBlock):
        # create zero matrix of appropriate dimensions
        mat = ZEROMAT
        for _ in range(nqubits - 1):
            mat = jnp.kron(mat, ZEROMAT)

        # perform matrix summation
        for b in block.blocks:
            mat = mat + block_to_jax(b, values, qubit_support, endianness=endianness)

    elif isinstance(block, ScaleBlock):
        (scale,) = _gate_parameters(block, values)
        if isinstance(scale, Tensor):
            scale = tensor_to_jnp(scale)
        mat = scale * block_to_jax(block.block, values, qubit_support, endianness=endianness)

    elif block.name in MAT_DICT.keys():
        block_mat = MAT_DICT[block.name]

        # add missing identities on unused qubits
        mat = _fill_identities(block_mat, block.qubit_support, qubit_support, endianness=endianness)

    elif isinstance(block, ProjectorBlock):
        from qadence.states import product_state

        bra = tensor_to_jnp(product_state(block.bra))
        ket = tensor_to_jnp(product_state(block.ket))

        block_mat = jnp.kron(ket, bra.T)

        mat = _fill_identities(
            block_mat,
            block.qubit_support,
            qubit_support,
            endianness=endianness,
        )
    else:
        raise TypeError(f"Conversion for block type {type(block)} not supported.")

    return mat
