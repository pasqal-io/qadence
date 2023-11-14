from __future__ import annotations

from sympy import cos, sin

from qadence.analog.device import RydbergDevice
from qadence.analog.interaction_hamiltonian import rydberg_interaction_hamiltonian
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


def add_background_hamiltonian(
    circuit: QuantumCircuit | AbstractBlock,
    device: RydbergDevice,
) -> QuantumCircuit | AbstractBlock:
    # Temporary check to allow both circuit or blocks as input
    is_circuit_input = isinstance(circuit, QuantumCircuit)
    target_block: AbstractBlock = circuit.block if is_circuit_input else circuit  # type: ignore

    # Create interaction hamiltonian:
    h_int = rydberg_interaction_hamiltonian(device)

    # Create addressing pattern:
    # h_addr = (...)

    h_background = h_int  # + h_addr

    block_parsed = apply_fn_to_blocks(
        target_block,
        _analog_to_hevo,
        device,
        h_background,
    )

    if is_circuit_input:
        return QuantumCircuit(device.register, block_parsed)
    else:
        return block_parsed


def _analog_to_hevo(
    block: AbstractBlock, device: RydbergDevice, h_background: AbstractBlock
) -> AbstractBlock:
    support = tuple(device.register.nodes)

    if isinstance(block, AnalogBlock):
        if isinstance(block, WaitBlock):
            # FIXME: Currently hardcoding the wait to be global
            duration = block.parameters.duration
            return HamEvo(h_background, duration / 1000)

        if isinstance(block, ConstantAnalogRotation):
            # FIXME: Currently hardcoding the rotations to be global
            duration = block.parameters.duration
            omega = block.parameters.omega
            delta = block.parameters.delta
            phase = block.parameters.phase

            x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
            z_terms = delta * add(N(i) for i in support)

            return HamEvo(x_terms - z_terms + h_background, duration / 1000)

    return block
