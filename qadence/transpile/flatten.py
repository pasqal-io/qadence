from __future__ import annotations

from copy import deepcopy
from functools import reduce, singledispatch
from typing import Generator, Type, overload

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    KronBlock,
    ScaleBlock,
)
from qadence.blocks.utils import _construct
from qadence.circuit import QuantumCircuit


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


@overload
def flatten(
    circuit: QuantumCircuit, types: list = [ChainBlock, KronBlock, AddBlock]
) -> QuantumCircuit:
    ...


@overload
def flatten(block: AbstractBlock, types: list = [ChainBlock, KronBlock, AddBlock]) -> AbstractBlock:
    ...


@singledispatch
def flatten(
    circ_or_block: AbstractBlock | QuantumCircuit, types: list = [ChainBlock, KronBlock, AddBlock]
) -> AbstractBlock | QuantumCircuit:
    raise NotImplementedError(f"digitalize is not implemented for {type(circ_or_block)}")


@flatten.register  # type: ignore[attr-defined]
def _(block: AbstractBlock, types: list = [ChainBlock, KronBlock, AddBlock]) -> AbstractBlock:
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


@flatten.register  # type: ignore[attr-defined]
def _(circuit: QuantumCircuit, types: list = [ChainBlock, KronBlock, AddBlock]) -> QuantumCircuit:
    return QuantumCircuit(circuit.n_qubits, flatten(circuit.block, types=types))
