from __future__ import annotations

from sympy import cos, sin

from qadence.analog.utils import rydberg_interaction_hamiltonian
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    AnalogBlock,
    ConstantAnalogRotation,
    Interaction,
    WaitBlock,
)
from qadence.blocks.utils import add
from qadence.circuit import QuantumCircuit
from qadence.operations import HamEvo, N, X, Y
from qadence.register import Register
from qadence.transpile import apply_fn_to_blocks


def add_interaction(
    circuit: QuantumCircuit | AbstractBlock,
    register: Register | None = None,
    interaction: Interaction = Interaction.NN,
    spacing: float = 1.0,
) -> QuantumCircuit:
    if isinstance(circuit, QuantumCircuit):
        register = circuit.register
        block = circuit.block
    elif register is not None:
        block = circuit
    else:
        raise ValueError("Provide a QuantumCircuit or block + register as input.")

    register = register._scale_positions(spacing)

    # Create interaction hamiltonian
    h_int = rydberg_interaction_hamiltonian(register, interaction)

    block_parsed = apply_fn_to_blocks(
        block,
        _add_interaction,
        register,
        h_int,
    )

    return QuantumCircuit(register, block_parsed)


def _add_interaction(
    block: AbstractBlock, register: Register, h_int: AbstractBlock
) -> AbstractBlock:
    support = tuple(register.nodes)

    if isinstance(block, AnalogBlock):
        if isinstance(block, WaitBlock):
            # Currently hardcoding the wait to be global
            duration = block.parameters.duration
            return HamEvo(h_int, duration / 1000)

        if isinstance(block, ConstantAnalogRotation):
            # Currently hardcoding the rotations to be global
            duration = block.parameters.duration
            omega = block.parameters.omega
            delta = block.parameters.delta
            phase = block.parameters.phase

            x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
            z_terms = delta * add(N(i) for i in support)

            return HamEvo(x_terms - z_terms + h_int, duration / 1000)

    return block
