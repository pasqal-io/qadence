from __future__ import annotations

from qadence.analog.hamiltonian_terms import (
    rydberg_drive_hamiltonian,
    rydberg_interaction_hamiltonian,
)
from qadence.blocks import chain
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    AnalogBlock,
    AnalogKron,
    ConstantAnalogRotation,
    WaitBlock,
)
from qadence.circuit import QuantumCircuit
from qadence.operations import HamEvo
from qadence.register import Register
from qadence.transpile import apply_fn_to_blocks


def add_background_hamiltonian(
    circuit: QuantumCircuit | AbstractBlock,
    register: Register | None = None,
) -> QuantumCircuit | AbstractBlock:
    # Temporary check to allow both circuit or blocks as input
    # Not sure yet if we want to support abstract blocks here, but
    # currently it's used for eigenvalue computation, which will
    # likely have to be refactored in another MR.

    is_circuit_input = isinstance(circuit, QuantumCircuit)

    if not is_circuit_input and register is None:
        raise ValueError("Block input requires an input to the `register` argument.")

    input_block: AbstractBlock = circuit.block if is_circuit_input else circuit  # type: ignore
    input_register: Register = circuit.register if is_circuit_input else register  # type: ignore

    if input_register.device_specs is not None:
        # Create interaction hamiltonian:
        h_int = rydberg_interaction_hamiltonian(input_register)

        # Create addressing pattern:
        # h_addr = (...)

        h_background = h_int  # + h_addr

        output_block = apply_fn_to_blocks(
            input_block,
            _analog_to_hevo,
            input_register,
            h_background,
        )
    else:
        output_block = input_block

    if is_circuit_input:
        return QuantumCircuit(input_register, output_block)
    else:
        return output_block


def _analog_to_hevo(
    block: AbstractBlock, register: Register, h_background: AbstractBlock
) -> AbstractBlock:
    if isinstance(block, AnalogBlock):
        if isinstance(block, WaitBlock):
            duration = block.parameters.duration
            return HamEvo(h_background, duration / 1000)

        if isinstance(block, ConstantAnalogRotation):
            h_drive = rydberg_drive_hamiltonian(block, register)
            duration = block.parameters.duration
            return HamEvo(h_drive + h_background, duration / 1000)

        if isinstance(block, AnalogKron):
            # FIXME: Clean this code
            ops = []
            for block in block.blocks:
                if isinstance(block, ConstantAnalogRotation):
                    duration = block.parameters.duration
                    h_drive = rydberg_drive_hamiltonian(block, register)
                    ops.append(HamEvo(h_drive + h_background, duration / 1000))
            if len(ops) == 0:
                duration = block.parameters.duration  # type: ignore
                ops.append(HamEvo(h_background, duration / 1000))
            return chain(*ops)

    return block
