from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import singledispatch
from typing import Any, overload

import numpy as np
from torch import Tensor, tensor

from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit


def reassign(block: AbstractBlock, qubit_map: dict[int, int]) -> AbstractBlock:
    """Update the support of a given block.

    Args:
        block (AbstractBlock): _description_
        qubit_map (dict[int, int]): _description_
    """
    from qadence.blocks import CompositeBlock, ControlBlock, ParametricControlBlock, ScaleBlock
    from qadence.blocks.utils import _construct

    def _block_with_updated_support(block: AbstractBlock) -> AbstractBlock:
        if isinstance(block, ControlBlock) or isinstance(block, ParametricControlBlock):
            old_qs = block.qubit_support
            new_control_block = deepcopy(block)
            new_control_block._qubit_support = tuple(qubit_map[i] for i in old_qs)
            (subblock,) = block.blocks
            new_control_block.blocks = (reassign(subblock, qubit_map),)  # type: ignore [assignment]
            return new_control_block
        elif isinstance(block, CompositeBlock):
            subblocks = tuple(_block_with_updated_support(b) for b in block.blocks)
            blk = _construct(type(block), subblocks)
            blk.tag = block.tag
            return blk
        elif isinstance(block, ScaleBlock):
            blk = deepcopy(block)  # type: ignore [assignment]
            blk.block = _block_with_updated_support(block.block)  # type: ignore [attr-defined]
            return blk
        else:
            blk = deepcopy(block)  # type: ignore [assignment]
            qs = tuple(qubit_map[i] for i in block.qubit_support)
            blk._qubit_support = qs  # type: ignore[attr-defined]
            return blk

    return _block_with_updated_support(block)


@overload
def invert_endianness(wf: Tensor) -> Tensor:
    ...


@overload
def invert_endianness(arr: np.ndarray) -> np.ndarray:
    ...


@overload
def invert_endianness(cntr: Counter) -> Counter:
    ...


@overload
def invert_endianness(cntrs: list) -> list:
    ...


@overload
def invert_endianness(circuit: QuantumCircuit, n_qubits: int) -> QuantumCircuit:
    ...


@overload
def invert_endianness(block: AbstractBlock, n_qubits: int, in_place: bool) -> AbstractBlock:
    ...


@singledispatch
def invert_endianness(
    x: QuantumCircuit | AbstractBlock | Tensor | Counter | np.ndarray, *args: Any
) -> QuantumCircuit | AbstractBlock | Tensor | Counter | np.ndarray:
    """Invert the endianness of a QuantumCircuit, AbstractBlock, wave function or Counter."""
    raise NotImplementedError(f"Unable to invert endianness of object {type(x)}.")


@invert_endianness.register(AbstractBlock)  # type: ignore[attr-defined]
def _(block: AbstractBlock, n_qubits: int = None, in_place: bool = False) -> AbstractBlock:
    if n_qubits is None:
        n_qubits = block.n_qubits
    """Flips endianness of the block"""
    if in_place:
        raise NotImplementedError
    bits = list(range(n_qubits))
    qubit_map = {i: j for (i, j) in zip(bits, reversed(bits))}
    return reassign(block, qubit_map=qubit_map)


@invert_endianness.register(Tensor)  # type: ignore[attr-defined]
def _(wf: Tensor) -> Tensor:
    """
    Inverts the endianness of a wave function.

    Args:
        wf (Tensor): the target wf as a torch Tensor of shape batch_size X 2**n_qubits

    Returns:
        The inverted wave function.
    """
    n_qubits = int(np.log2(wf.shape[1]))
    ls = list(range(2**n_qubits))
    permute_ind = tensor([int(f"{num:0{n_qubits}b}"[::-1], 2) for num in ls])
    return wf[:, permute_ind]


@invert_endianness.register(np.ndarray)  # type: ignore[attr-defined]
def _(arr: np.ndarray) -> np.ndarray:
    return invert_endianness(tensor(arr)).numpy()


@invert_endianness.register(Counter)  # type: ignore[attr-defined]
def _(cntr: Counter) -> Counter:
    return Counter(
        {
            format(int(bstring[::-1], 2), "0{}b".format(len(bstring))): count
            for bstring, count in cntr.items()
        }
    )


@invert_endianness.register(list)  # type: ignore[attr-defined]
def _(cntrs: list) -> list:
    return list(map(invert_endianness, cntrs))


@invert_endianness.register(QuantumCircuit)  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit) -> QuantumCircuit:
    """This method inverts a circuit "vertically".

    All gates are same but qubit indices are ordered inversely,
    such that bitstrings 00111 become 11100 when measured. Handy for
    big-endian <> little-endian conversion

    Returns:
        QuantumCircuit with endianess switched
    """
    return QuantumCircuit(
        circuit.n_qubits, invert_endianness(circuit.block, circuit.n_qubits, False)
    )
