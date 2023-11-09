from __future__ import annotations

from functools import reduce
from typing import Callable, TypeVar, overload

from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit

BlockOrCirc = TypeVar("BlockOrCirc", AbstractBlock, QuantumCircuit)


@overload
def transpile(
    *fs: Callable[[AbstractBlock], AbstractBlock]
) -> Callable[[AbstractBlock], AbstractBlock]:
    ...


@overload
def transpile(
    *fs: Callable[[QuantumCircuit], QuantumCircuit]
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    ...


def transpile(*fs: Callable) -> Callable:
    """`AbstractBlock` or `QuantumCircuit` transpilation.

    Compose functions that
    accept a circuit/block and returns a circuit/block.

    Arguments:
        *fs: composable functions that either map blocks to blocks
            (`Callable[[AbstractBlock], AbstractBlock]`)
            or circuits to circuits (`Callable[[QuantumCircuit], QuantumCircuit]`).

    Returns:
        Composed function.

    Examples:

    Flatten a block of nested chains and krons:
    ```python exec="on" source="material-block" result="json"
    from qadence import *
    from qadence.transpile import transpile, flatten, scale_primitive_blocks_only

    b = chain(2 * chain(chain(X(0), Y(0))), kron(kron(X(0), X(1))))
    print(b)
    print() # markdown-exec: hide

    # both flatten and scale_primitive_blocks_only are functions that accept and
    # return a block
    t = transpile(flatten, scale_primitive_blocks_only)(b)
    print(t)
    ```

    We also proved a decorator to easily turn a function `Callable[[AbstractBlock], AbstractBlock]`
    into a `Callable[[QuantumCircuit], QuantumCircuit]` to be used in circuit transpilation.
    ```python exec="on" source="material-block" result="json"
    from qadence import *
    from qadence.transpile import transpile, blockfn_to_circfn, flatten

    # We want to pass this circuit to `transpile` instead of a block,
    # so we need functions that map from a circuit to a circuit.
    circ = QuantumCircuit(2, chain(chain(X(0), chain(X(1)))))

    @blockfn_to_circfn
    def fn(block):
        # un-decorated function accepts a block and returns a block
        return block * block

    transp = transpile(
        # the decorated function accepts a circuit and returns a circuit
        fn,
        # already existing functions can also be decorated
        blockfn_to_circfn(flatten)
    )
    print(transp(circ))
    ```
    """
    return lambda x: reduce(lambda acc, f: f(acc), reversed(fs), x)


def blockfn_to_circfn(
    fn: Callable[[AbstractBlock], AbstractBlock]
) -> Callable[[QuantumCircuit], QuantumCircuit]:
    return lambda circ: QuantumCircuit(circ.register, fn(circ.block))
