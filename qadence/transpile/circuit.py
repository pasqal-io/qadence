from __future__ import annotations

from typing import List

from qadence.blocks.utils import chain
from qadence.circuit import QuantumCircuit
from qadence.operations import I


def reverse_program(circuit: QuantumCircuit, inplace: bool = True) -> QuantumCircuit:
    raise (NotImplementedError("Must also implement at Block level"))


def parametric_shift_rule(circuit: QuantumCircuit) -> List[QuantumCircuit]:
    raise NotImplementedError


def fill_identities(circ: QuantumCircuit) -> QuantumCircuit:
    empty_wires = set(range(circ.n_qubits)) - set(circ.block.qubit_support)
    if len(empty_wires) > 0:
        ids = chain(I(i) for i in empty_wires)
        return QuantumCircuit(circ.n_qubits, chain(circ.block, ids))
    return circ
