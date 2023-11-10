from __future__ import annotations

from typing import Any, Callable

from qadence.blocks import AbstractBlock, CompositeBlock, add, chain, kron
from qadence.blocks.analog import AnalogComposite

COMPOSE_FN_DICT = {
    "ChainBlock": chain,
    "AnalogChain": chain,
    "KronBlock": kron,
    "AnalogKron": kron,
    "AddBlock": add,
}


def apply_fn_to_blocks(
    input_block: AbstractBlock, block_fn: Callable, **kwargs: Any
) -> AbstractBlock:
    """
    Recurses through the block tree and applies a given function to all the leaf blocks.

    Arguments:
        input_block: tree of blocks on which to apply the recurse.
        block_fn: callable function to apply to each leaf block.
        kwargs: any keyword arguments to pass to the leaf function.
    """

    if isinstance(input_block, (CompositeBlock, AnalogComposite)):
        parsed_blocks = [
            apply_fn_to_blocks(block, block_fn, **kwargs) for block in input_block.blocks
        ]
        compose_fn = COMPOSE_FN_DICT[type(input_block).__name__]
        output_block = compose_fn(*parsed_blocks)  # type: ignore [arg-type]
    else:
        output_block = block_fn(input_block, **kwargs)

    return output_block
