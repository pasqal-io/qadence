from __future__ import annotations

from typing import Generator, List, Type, TypeVar, Union

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    KronBlock,
    PrimitiveBlock,
    PutBlock,
)
from qadence.blocks.analog import AnalogBlock, AnalogComposite
from qadence.blocks.analog import chain as analog_chain
from qadence.blocks.analog import kron as analog_kron
from qadence.logger import get_logger

logger = get_logger(__name__)


TPrimitiveBlock = TypeVar("TPrimitiveBlock", bound=PrimitiveBlock)
TCompositeBlock = TypeVar("TCompositeBlock", bound=CompositeBlock)


def _construct(
    Block: Type[TCompositeBlock],
    args: tuple[Union[AbstractBlock, Generator, List[AbstractBlock]], ...],
) -> TCompositeBlock:
    if len(args) == 1 and isinstance(args[0], Generator):
        args = tuple(args[0])
    return Block([b for b in args])  # type: ignore [arg-type]


def chain(*args: Union[AbstractBlock, Generator, List[AbstractBlock]]) -> ChainBlock:
    """Chain blocks sequentially. On digital backends this can be interpreted
    loosely as a matrix mutliplication of blocks. In the analog case it chains
    blocks in time.

    Arguments:
        *args: Blocks to chain. Can also be a generator.

    Returns:
        ChainBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, chain

    b = chain(X(0), Y(0))

    # or use a generator
    b = chain(X(i) for i in range(3))
    print(b)
    ```
    """
    # ugly hack to use `AnalogChain` if we are dealing only with analog blocks
    if len(args) and all(
        isinstance(a, AnalogBlock) or isinstance(a, AnalogComposite) for a in args
    ):
        return analog_chain(*args)  # type: ignore[return-value,arg-type]
    return _construct(ChainBlock, args)


def kron(*args: Union[AbstractBlock, Generator]) -> KronBlock:
    """Stack blocks vertically. On digital backends this can be intepreted
    loosely as a kronecker product of blocks. In the analog case it executes
    blocks parallel in time.

    Arguments:
        *args: Blocks to kron. Can also be a generator.

    Returns:
        KronBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, kron

    b = kron(X(0), Y(1))

    # or use a generator
    b = kron(X(i) for i in range(3))
    print(b)
    ```
    """
    # ugly hack to use `AnalogKron` if we are dealing only with analog blocks
    if len(args) and all(
        isinstance(a, AnalogBlock) or isinstance(a, AnalogComposite) for a in args
    ):
        return analog_kron(*args)  # type: ignore[return-value,arg-type]
    return _construct(KronBlock, args)


def add(*args: Union[AbstractBlock, Generator]) -> AddBlock:
    """Sums blocks.

    Arguments:
        *args: Blocks to add. Can also be a generator.

    Returns:
        AddBlock

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X, Y, add

    b = add(X(0), Y(0))

    # or use a generator
    b = add(X(i) for i in range(3))
    print(b)
    ```
    """
    return _construct(AddBlock, args)


def tag(block: AbstractBlock, tag: str) -> AbstractBlock:
    block.tag = tag
    return block


def put(block: AbstractBlock, min_qubit: int, max_qubit: int) -> PutBlock:
    from qadence.transpile import reassign

    support = tuple(range(min(block.qubit_support), max(block.qubit_support) + 1))
    shifted_block = reassign(block, {i: i - min(support) for i in support})
    return PutBlock(shifted_block, tuple(range(min_qubit, max_qubit + 1)))
