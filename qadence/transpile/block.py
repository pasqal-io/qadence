from __future__ import annotations

from copy import deepcopy
from functools import reduce, singledispatch
from typing import Callable, Generator, Iterable, Type

import sympy

from qadence import operations
from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    AnalogBlock,
    ChainBlock,
    CompositeBlock,
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
from qadence.parameters import Parameter


def _flat_blocks(block: AbstractBlock, T: Type) -> Generator:
    """Constructs a generator that flattens nested `CompositeBlock`s of type `T`.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence.transpile.block import _flat_blocks
    from qadence.blocks import ChainBlock
    from qadence import chain, X

    x = chain(chain(chain(X(0)), X(0)))
    assert tuple(_flat_blocks(x, ChainBlock)) == (X(0), X(0))
    ```
    """
    if isinstance(block, T):
        # here we do the flattening
        for b in block.blocks:
            if isinstance(b, T):
                yield from _flat_blocks(b, T)
            else:
                yield flatten(b, [T])
    elif isinstance(block, CompositeBlock):
        # here we make sure that we don't get stuck at e.g. `KronBlock`s if we
        # want to flatten `ChainBlock`s
        yield from (flatten(b, [T]) for b in block.blocks)
    elif isinstance(block, ScaleBlock):
        blk = deepcopy(block)
        blk.block = flatten(block.block, [T])
        yield blk
    else:
        yield block


def flatten(block: AbstractBlock, types: list = [ChainBlock, KronBlock, AddBlock]) -> AbstractBlock:
    """Flattens the given types of `CompositeBlock`s if possible.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import chain, kron, X
    from qadence.transpile import flatten
    from qadence.blocks import ChainBlock, KronBlock, AddBlock

    x = chain(chain(chain(X(0))), kron(kron(X(0))))

    # flatten only `ChainBlock`s
    assert flatten(x, [ChainBlock]) == chain(X(0), kron(kron(X(0))))

    # flatten `ChainBlock`s and `KronBlock`s
    assert flatten(x, [ChainBlock, KronBlock]) == chain(X(0), kron(X(0)))

    # flatten `AddBlock`s (does nothing in this case)
    assert flatten(x, [AddBlock]) == x
    ```
    """
    if isinstance(block, CompositeBlock):

        def fn(b: AbstractBlock, T: Type) -> AbstractBlock:
            return _construct(type(block), tuple(_flat_blocks(b, T)))

        return reduce(fn, types, block)  # type: ignore[arg-type]
    elif isinstance(block, ScaleBlock):
        blk = deepcopy(block)
        blk.block = flatten(block.block, types=types)
        return blk
    else:
        return block


def repeat(
    Block: Type[TPrimitiveBlock], support: Iterable[int], parameter: str | Parameter | None = None
) -> KronBlock:
    if parameter is None:
        return kron(Block(i) for i in support)  # type: ignore [arg-type]
    return kron(Block(i, parameter) for i in support)  # type: ignore [call-arg, arg-type]


def set_trainable(
    blocks: AbstractBlock | list[AbstractBlock], value: bool = True, inplace: bool = True
) -> AbstractBlock | list[AbstractBlock]:
    """Set the trainability of all parameters in a block to a given value

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
    """Moves a block from global to local qubit numbers by adding PutBlocks and reassigning
    qubit locations approriately.

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

    if isinstance(block, operations.ControlBlock):
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
    """When given a scaled CompositeBlock consisting of several PrimitiveBlocks,
    move the scale all the way down into the leaves of the block tree.

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
