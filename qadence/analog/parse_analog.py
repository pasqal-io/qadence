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
from qadence.register import Register
from qadence.transpile import apply_fn_to_blocks


def add_background_hamiltonian(
    circuit: QuantumCircuit | AbstractBlock,
    device: RydbergDevice | None = None,
    register: Register | None = None,
) -> QuantumCircuit | AbstractBlock:
    # Temporary check to allow both circuit or blocks as input
    # Not sure yet if we want to support abstract blocks here, but
    # currently it's used for eigenvalue computation, which will
    # likely have to be refactored in another MR.

    is_circuit_input = isinstance(circuit, QuantumCircuit)
    target_block: AbstractBlock = circuit.block if is_circuit_input else circuit  # type: ignore
    target_register: Register = circuit.register if is_circuit_input else register  # type: ignore

    if not is_circuit_input and register is None:
        raise ValueError("Block input requires an input to the `register` argument.")

    if device is None:
        device = RydbergDevice()

    # Create interaction hamiltonian:
    h_int = rydberg_interaction_hamiltonian(target_register, device)

    # Create addressing pattern:
    # h_addr = (...)

    h_background = h_int  # + h_addr

    block_parsed = apply_fn_to_blocks(
        target_block,
        _analog_to_hevo,
        target_register,
        h_background,
    )

    if is_circuit_input:
        return QuantumCircuit(target_register, block_parsed)
    else:
        return block_parsed


def _analog_to_hevo(
    block: AbstractBlock, register: Register, h_background: AbstractBlock
) -> AbstractBlock:
    support = tuple(register.nodes)

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
