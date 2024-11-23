from __future__ import annotations

from qadence.backend import ConvertedCircuit
from qadence.blocks.abstract import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.noise.protocols import NoiseHandler

from .apply_fn import apply_fn_to_blocks


def _set_noise(
    block: AbstractBlock,
    noise: NoiseHandler | None,
    target_class: type[AbstractBlock] | None = None,
) -> AbstractBlock:
    """Changes the noise protocol of a given block in place."""
    if target_class is not None:
        if isinstance(block, target_class):
            block._noise = noise  # type: ignore [attr-defined]
    else:
        block._noise = noise  # type: ignore [attr-defined]

    return block


def set_noise(
    circuit: QuantumCircuit | AbstractBlock | ConvertedCircuit,
    noise: NoiseHandler | None,
    target_class: AbstractBlock | None = None,
) -> QuantumCircuit | AbstractBlock:
    """
    Parses a `QuantumCircuit` or `CompositeBlock` to add noise to specific gates.

    If `circuit` is a `ConvertedCircuit`, this is done within `circuit.abstract`.

    Changes the input in place.

    Arguments:
        circuit: the circuit or block to parse.
        noise: the NoiseHandler protocol to change to, or `None` to remove the noise.
        target_class: optional class to selectively add noise to.
    """
    input_block: AbstractBlock
    if isinstance(circuit, ConvertedCircuit):
        input_block = circuit.abstract.block
    elif isinstance(circuit, QuantumCircuit):
        input_block = circuit.block
    else:
        input_block = circuit

    apply_fn_to_blocks(input_block, _set_noise, noise, target_class)

    return circuit
