from __future__ import annotations

from typing import Any, Callable

from qadence.blocks import AbstractBlock, CompositeBlock, add, chain, kron
from qadence.blocks.analog import AnalogComposite
from qadence.circuit import QuantumCircuit

COMPOSE_FN_DICT = {
    "ChainBlock": chain,
    "AnalogChain": chain,
    "KronBlock": kron,
    "AnalogKron": kron,
    "AddBlock": add,
}


def apply_fn_to_blocks(
    input_block: QuantumCircuit | AbstractBlock, block_fn: Callable, **kwargs: Any
) -> QuantumCircuit | AbstractBlock:
    """
    Recurses through the block tree and applies a given function to all the leaf blocks.

    Arguments:
        input_block: block or circuit on which to apply the function.
        block_fn: callable function to apply to each leaf block.
        kwargs: any keyword arguments to pass to the leaf function.
    """

    work_block = input_block.block if isinstance(input_block, QuantumCircuit) else input_block

    if isinstance(work_block, (CompositeBlock, AnalogComposite)):
        parsed_blocks = [
            apply_fn_to_blocks(block, block_fn, **kwargs) for block in work_block.blocks
        ]
        compose_fn = COMPOSE_FN_DICT[type(work_block).__name__]
        output_block = compose_fn(*parsed_blocks)  # type: ignore [arg-type]
    else:
        output_block = block_fn(work_block, **kwargs)

    if isinstance(input_block, QuantumCircuit):
        output_block = QuantumCircuit(input_block.register, output_block)  # type: ignore [assignment]

    return output_block
