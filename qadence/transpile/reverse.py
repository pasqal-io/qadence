from __future__ import annotations

from functools import singledispatch
from typing import Any, overload

from qadence import QuantumCircuit
from qadence.blocks import AbstractBlock, ChainBlock, chain


@overload
def reverse(circuit: QuantumCircuit, in_place: bool) -> QuantumCircuit:
    ...


@overload
def reverse(block: AbstractBlock, in_place: bool) -> AbstractBlock:
    ...


@singledispatch
def reverse(x: QuantumCircuit | AbstractBlock, *args: Any) -> QuantumCircuit | AbstractBlock:
    """Reverses a QuantumCircuit or AbstractBlock."""
    raise NotImplementedError(f"Unable to invert endianness of object {type(x)}.")


@reverse.register(AbstractBlock)  # type: ignore[attr-defined]
def _(block: AbstractBlock, in_place: bool = False) -> AbstractBlock:
    """Reverses a block if its a ChainBlock."""
    if in_place:
        raise NotImplementedError
    if isinstance(block, ChainBlock):
        return chain(*(reverse(b, in_place=in_place) for b in reversed(block.blocks)))
    else:
        return block


@reverse.register(QuantumCircuit)  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit) -> QuantumCircuit:
    """This method reverses a circuit.

    Returns:
        A reversed QuantumCircuit.
    """
    return QuantumCircuit(circuit.n_qubits, reverse(circuit.block, False))
