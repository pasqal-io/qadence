from __future__ import annotations

from functools import singledispatch
from typing import overload

from qadence import QuantumCircuit
from qadence.blocks import AbstractBlock, ChainBlock, chain, tag


@overload
def reverse(circuit: QuantumCircuit) -> QuantumCircuit:
    ...


@overload
def reverse(block: AbstractBlock) -> AbstractBlock:
    ...


@singledispatch
def reverse(x: QuantumCircuit | AbstractBlock) -> QuantumCircuit | AbstractBlock:
    """Reverses a QuantumCircuit or AbstractBlock."""
    raise NotImplementedError(f"Unable to invert endianness of object {type(x)}.")


@reverse.register(AbstractBlock)  # type: ignore[attr-defined]
def _(block: AbstractBlock) -> AbstractBlock:
    """Reverses a block if its a ChainBlock."""
    if isinstance(block, ChainBlock):
        blk = chain(*(reverse(b) for b in reversed(block.blocks)))
        return tag(blk, block.tag) if block.tag is not None else blk
    else:
        return block


@reverse.register(QuantumCircuit)  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit) -> QuantumCircuit:
    """This method reverses a circuit.

    Returns:
        A reversed QuantumCircuit.
    """
    return QuantumCircuit(circuit.n_qubits, reverse(circuit.block))
