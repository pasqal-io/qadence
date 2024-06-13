from __future__ import annotations

from copy import deepcopy
from functools import singledispatch
from logging import getLogger
from typing import Callable, Iterable, Type

import sympy

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    AnalogBlock,
    ChainBlock,
    CompositeBlock,
    ControlBlock,
    KronBlock,
    PrimitiveBlock,
    PutBlock,
    ScaleBlock,
    add,
    chain,
    kron,
)
from qadence.blocks.utils import (
    TPrimitiveBlock,
    _construct,
    parameters,
)
from qadence.operations import SWAP, I
from qadence.parameters import Parameter

logger = getLogger(__name__)


def repeat(
    Block: Type[TPrimitiveBlock], support: Iterable[int], parameter: str | Parameter | None = None
) -> KronBlock:
    if parameter is None:
        return kron(Block(i) for i in support)  # type: ignore [arg-type]
    return kron(Block(i, parameter) for i in support)  # type: ignore [call-arg, arg-type]


def set_trainable(
    blocks: AbstractBlock | list[AbstractBlock], value: bool = True, inplace: bool = True
) -> AbstractBlock | list[AbstractBlock]:
    """Set the trainability of all parameters in a block to a given value.

    Args:
        blocks (AbstractBlock | list[AbstractBlock]): Block or list of blocks for which
            to set the trainable attribute
        value (bool, optional): The value of the trainable attribute to assign to the input blocks
        inplace (bool, optional): Whether to modify the block(s) in place or not. Currently, only

    Raises:
        NotImplementedError: if the `inplace` argument is set to False, the function will
            raise  this exception

    Returns:
        AbstractBlock | list[AbstractBlock]: the input block or list of blocks with the trainable
            attribute set to the given value
    """

    if isinstance(blocks, AbstractBlock):
        blocks = [blocks]

    if inplace:
        for block in blocks:
            params: list[sympy.Basic] = parameters(block)
            for p in params:
                if not p.is_number:
                    p.trainable = value
    else:
        raise NotImplementedError("Not inplace set_trainable is not yet available")

    return blocks if len(blocks) > 1 else blocks[0]


def validate(block: AbstractBlock) -> AbstractBlock:
    """Moves a block from global to local qubit numbers by adding PutBlocks.

    Reassigns qubit locations appropriately.

    # Example
    ```python exec="on" source="above" result="json"
    from qadence.blocks import chain
    from qadence.operations import X
    from qadence.transpile import validate

    x = chain(chain(X(0)), chain(X(1)))
    print(x)
    print(validate(x))
    ```
    """
    vblock: AbstractBlock
    from qadence.transpile import reassign

    if isinstance(block, ControlBlock):
        vblock = deepcopy(block)
        b: AbstractBlock
        (b,) = block.blocks
        b = reassign(b, {i: i - min(b.qubit_support) for i in b.qubit_support})
        b = validate(b)
        vblock.blocks = (b,)  # type: ignore[assignment]

    elif isinstance(block, CompositeBlock):
        blocks = []
        for b in block.blocks:
            mi, ma = min(b.qubit_support), max(b.qubit_support)
            nb = reassign(b, {i: i - min(b.qubit_support) for i in b.qubit_support})
            nb = validate(nb)
            nb = PutBlock(nb, tuple(range(mi, ma + 1)))
            blocks.append(nb)
        try:
            vblock = _construct(type(block), tuple(blocks))
        except AssertionError as e:
            if str(e) == "Make sure blocks act on distinct qubits!":
                vblock = chain(*blocks)
            else:
                raise e

    elif isinstance(block, PrimitiveBlock):
        vblock = deepcopy(block)

    else:
        raise NotImplementedError

    vblock.tag = block.tag
    return vblock


@singledispatch
def scale_primitive_blocks_only(block: AbstractBlock, scale: sympy.Basic = None) -> AbstractBlock:
    """Push the scale all the way down into the leaves of the block tree.

    When given a scaled CompositeBlock consisting of several PrimitiveBlocks.

    Arguments:
        block: The block to be transpiled.
        scale: An optional scale parameter. Only to be used for recursive calls internally.

    Returns:
        AbstractBlock: A block of the same type where the scales have been moved into the subblocks.

    Examples:

    There are two different cases:
    `ChainBlock`s/`KronBlock`s: Only the first subblock needs to be scaled because chains/krons
    represent multiplications.
    ```python exec="on" source="above" result="json"
    from qadence import chain, X, RX
    from qadence.transpile import scale_primitive_blocks_only
    b = 2 * chain(X(0), RX(0, "theta"))
    print(b)
    # After applying scale_primitive_blocks_only
    print(scale_primitive_blocks_only(b))
    ```

    `AddBlock`s: Consider 2 * add(X(0), RX(0, "theta")).  The scale needs to be added to all
    subblocks.  We get add(2 * X(0), 2 * RX(0, "theta")).
    ```python exec="on" source="above" result="json"
    from qadence import add, X, RX
    from qadence.transpile import scale_primitive_blocks_only
    b = 2 * add(X(0), RX(0, "theta"))
    print(b)
    # After applying scale_primitive_blocks_only
    print(scale_primitive_blocks_only(b))
    ```
    """
    raise NotImplementedError(f"scale_primitive_blocks_only is not implemented for {type(block)}")


@scale_primitive_blocks_only.register
def _(block: ScaleBlock, scale: sympy.Basic = None) -> AbstractBlock:
    (scale2,) = block.parameters.expressions()
    s = scale2 if scale is None else scale * scale2
    blk = scale_primitive_blocks_only(block.block, s)
    blk.tag = block.tag
    return blk


@scale_primitive_blocks_only.register
def _(block: ChainBlock, scale: sympy.Basic = None) -> CompositeBlock:
    blk = scale_only_first_block(chain, block, scale)
    blk.tag = block.tag
    return blk


@scale_primitive_blocks_only.register
def _(block: KronBlock, scale: sympy.Basic = None) -> CompositeBlock:
    blk = scale_only_first_block(kron, block, scale)
    blk.tag = block.tag
    return blk


@scale_primitive_blocks_only.register
def _(block: AddBlock, scale: sympy.Basic = None) -> CompositeBlock:
    blk = add(scale_primitive_blocks_only(b, scale) for b in block.blocks)
    blk.tag = block.tag
    return blk


@scale_primitive_blocks_only.register
def _(block: PrimitiveBlock, scale: sympy.Basic = None) -> AbstractBlock:
    if scale is None:
        return block
    b: ScaleBlock = block * scale
    return b


@scale_primitive_blocks_only.register
def _(block: AnalogBlock, scale: sympy.Basic = None) -> AbstractBlock:
    if scale is not None:
        raise NotImplementedError("Cannot scale `AnalogBlock`s!")
    return block


def scale_only_first_block(
    fn: Callable, block: CompositeBlock, scale: sympy.Basic = None
) -> CompositeBlock:
    if len(block.blocks):
        first, rest = block.blocks[0], block.blocks[1:]
        firstscaled = scale_primitive_blocks_only(first, scale)

        blk: CompositeBlock
        blk = fn(firstscaled, *[scale_primitive_blocks_only(b, None) for b in rest])
        return blk
    else:
        return block


@singledispatch
def fill_identities(block: AbstractBlock, start: int, stop: int) -> AbstractBlock:
    return block


@fill_identities.register
def _(block: PrimitiveBlock, start: int, stop: int) -> AbstractBlock:
    full_support = tuple(range(start, stop + 1))
    if block.qubit_support == full_support:
        return block
    tag = block.tag
    block.tag = None
    bs = [block] + [I(i) for i in (set(full_support) - set(block.qubit_support))]
    b = kron(*sorted(bs, key=lambda x: x.qubit_support))
    b.tag = tag
    return b


@fill_identities.register
def _(block: SWAP, start: int, stop: int) -> AbstractBlock:
    full_support = tuple(range(start, stop + 1))
    if block.qubit_support == full_support:
        return block
    tag = block.tag
    block.tag = None
    bs = [block] + [chain(I(i), I(i)) for i in (set(full_support) - set(block.qubit_support))]
    b = kron(*sorted(bs, key=lambda x: x.qubit_support))
    b.tag = tag
    return b


@fill_identities.register
def _(block: ChainBlock, start: int, stop: int) -> AbstractBlock:
    b = chain(fill_identities(b, start, stop) for b in block)
    b.tag = block.tag
    return b


def _fill_kron(block: KronBlock, start: int, stop: int) -> list[AbstractBlock]:
    def length(b: AbstractBlock | list) -> int:
        if isinstance(b, list):
            return max(map(length, b))
        elif isinstance(b, ChainBlock):
            return len(b)
        elif isinstance(b, SWAP):
            return 2
        else:
            return 1

    def append_ids(block: AbstractBlock, total: int) -> AbstractBlock:
        qs = block.qubit_support
        ids = [I(i) for i in range(min(qs), max(qs)) for _ in range(length(block), total)]
        bs = [block] + ids
        return chain(*bs)

    def id_chain(i: int, max_len: int) -> AbstractBlock:
        return chain(I(i) for _ in range(max_len))

    # fill subblocks with identities
    bs = [fill_identities(b, min(b.qubit_support), max(b.qubit_support)) for b in block]

    # make all subblocks equally "long" horizontally
    max_len = length(bs)
    bs = [append_ids(b, max_len) for b in bs]

    # fill potentially completely empty wires
    bs += [id_chain(i, max_len) for i in (set(range(start, stop + 1)) - set(block.qubit_support))]

    return sorted(bs, key=lambda x: x.qubit_support)


@fill_identities.register
def _(block: KronBlock, start: int, stop: int) -> AbstractBlock:
    b = kron(*_fill_kron(block, start, stop))  # type: ignore[misc]
    b.tag = block.tag
    return b


@fill_identities.register
def _(block: AddBlock, start: int, stop: int) -> AbstractBlock:
    b = add(fill_identities(b, start, stop) for b in block)
    b.tag = block.tag
    return b


def is_kron_of_primitives(block: AbstractBlock) -> bool:
    return isinstance(block, (KronBlock)) and all(
        [isinstance(b, PrimitiveBlock) and b.n_supports == 1 for b in block.blocks]
    )


def is_chain_of_primitivekrons(block: AbstractBlock) -> bool:
    if isinstance(block, ChainBlock) and len(block) > 1:
        same_length = len(set(map(len, block))) == 1
        kron_of_primitives_same_qubit = all(
            is_kron_of_primitives(b) and b.qubit_support == block.qubit_support
            for b in block.blocks
        )
        return same_length and kron_of_primitives_same_qubit
    else:
        return False


def chain_single_qubit_ops(block: AbstractBlock) -> AbstractBlock:
    """Transpile a chain of krons into a kron of chains of single qubit operations.

    Examples:
    ```python exec="on" source="above" result="json"
    from qadence import hea
    from qadence.transpile.block import chain_single_qubit_ops

    # Consider a single HEA layer
    block = hea(2,1)
    print(block)

    # After applying chain_single_qubit_ops, we get:
    print(chain_single_qubit_ops(block))
    ```
    """
    if is_chain_of_primitivekrons(block):
        try:
            return kron(*map(lambda bs: chain(*bs), zip(*block)))  # type: ignore[misc]
        except Exception as e:
            logger.debug(
                f"Unable to transpile {block} using chain_single_qubit_ops\
                         due to {e}. Returning original circuit."
            )
            return block

    elif isinstance(block, CompositeBlock):
        return _construct(type(block), tuple(chain_single_qubit_ops(b) for b in block.blocks))
    else:
        return block
