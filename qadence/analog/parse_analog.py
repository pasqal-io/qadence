from __future__ import annotations

from qadence.analog.hamiltonian_terms import (
    rydberg_drive_hamiltonian,
    rydberg_interaction_hamiltonian,
    rydberg_pattern_hamiltonian,
)
from qadence.blocks import chain
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
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
    """
    Parses a `QuantumCircuit` to transform `AnalogBlocks` to `HamEvo`.

    Depends on the circuit `Register` and included `RydbergDevice` specifications.

    Currently checks if input is either circuit or block and adjusts
    the ouput accordingly. Running this function on single blocks is
    currently used for eigenvalue computation for GPSR.

    Arguments:
        circuit: the circuit to parse, or single block to transform.
        register: needed for calling the function on a single block.
    """
    # FIXME: revisit eigenvalues of analog blocks and clean this code.

    is_circuit_input = isinstance(circuit, QuantumCircuit)

    if not is_circuit_input and register is None:
        raise ValueError("Block input requires an input to the `register` argument.")

    input_block: AbstractBlock = circuit.block if is_circuit_input else circuit  # type: ignore
    input_register: Register = circuit.register if is_circuit_input else register  # type: ignore

    if input_register.device_specs is not None:
        # Create interaction hamiltonian:
        h_int = rydberg_interaction_hamiltonian(input_register)

        # Create addressing pattern:
        h_addr = rydberg_pattern_hamiltonian(input_register)

        output_block = apply_fn_to_blocks(
            input_block,
            _analog_to_hevo,
            input_register,
            (h_int, h_addr),
        )
    else:
        output_block = input_block

    if is_circuit_input:
        return QuantumCircuit(input_register, output_block)
    else:
        return output_block


def _build_ham_evo(
    block: WaitBlock | ConstantAnalogRotation,
    h_int: AbstractBlock,
    h_drive: AbstractBlock | None,
    h_addr: AbstractBlock | None,
) -> HamEvo:
    duration = block.parameters.duration
    h_block = h_int
    if h_drive is not None:
        h_block += h_drive
    if block.add_pattern and h_addr is not None:
        h_block += h_addr
    return HamEvo(h_block, duration / 1000)


def _analog_to_hevo(
    block: AbstractBlock,
    register: Register,
    h_terms: tuple[AbstractBlock, AbstractBlock | None],
) -> AbstractBlock:
    """
    Converter from AnalogBlock to the respective HamEvo.

    Any other block not covered by the specific conditions below is left unchanged.
    """

    h_int, h_addr = h_terms

    if isinstance(block, WaitBlock):
        return _build_ham_evo(block, h_int, None, h_addr)

    if isinstance(block, ConstantAnalogRotation):
        h_drive = rydberg_drive_hamiltonian(block, register)
        return _build_ham_evo(block, h_int, h_drive, h_addr)

    if isinstance(block, AnalogKron):
        # Needed to ensure kronned Analog blocks are implemented
        # in sequence, consistent with the current Pulser implementation.
        # FIXME: Revisit this assumption and the need for AnalogKron to have
        # the same duration, and clean this code accordingly.
        # https://github.com/pasqal-io/qadence/issues/226
        ops = []
        for block in block.blocks:
            if isinstance(block, ConstantAnalogRotation):
                h_drive = rydberg_drive_hamiltonian(block, register)
                ops.append(_build_ham_evo(block, h_int, h_drive, h_addr))
        if len(ops) == 0:
            ops.append(_build_ham_evo(block, h_int, None, h_addr))  # type: ignore [arg-type]
        return chain(*ops)

    return block
