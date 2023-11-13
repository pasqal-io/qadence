from __future__ import annotations

from sympy import cos, sin

from qadence.analog.device import RydbergDevice
from qadence.analog.utils import rydberg_interaction_hamiltonian
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    AnalogBlock,
    ConstantAnalogRotation,
    WaitBlock,
)
from qadence.blocks.utils import add
from qadence.circuit import QuantumCircuit
from qadence.operations import HamEvo, N, X, Y
from qadence.transpile import apply_fn_to_blocks


def add_interaction(
    circuit: QuantumCircuit,
    device: RydbergDevice,
) -> QuantumCircuit:
    # Create interaction hamiltonian
    h_int = rydberg_interaction_hamiltonian(device)

    block_parsed = apply_fn_to_blocks(
        circuit.block,
        _add_interaction,
        device,
        h_int,
    )

    return QuantumCircuit(device.register, block_parsed)


def _add_interaction(
    block: AbstractBlock, device: RydbergDevice, h_int: AbstractBlock
) -> AbstractBlock:
    support = tuple(device.register.nodes)

    if isinstance(block, AnalogBlock):
        if isinstance(block, WaitBlock):
            # FIXME: Currently hardcoding the wait to be global
            duration = block.parameters.duration
            return HamEvo(h_int, duration / 1000)

        if isinstance(block, ConstantAnalogRotation):
            # FIXME: Currently hardcoding the rotations to be global
            duration = block.parameters.duration
            omega = block.parameters.omega
            delta = block.parameters.delta
            phase = block.parameters.phase

            x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
            z_terms = delta * add(N(i) for i in support)

            return HamEvo(x_terms - z_terms + h_int, duration / 1000)

    return block
