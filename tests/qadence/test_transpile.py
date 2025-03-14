from __future__ import annotations

from qadence.blocks import AbstractBlock, AddBlock, ChainBlock, KronBlock
from qadence.blocks.utils import chain, kron
from qadence.constructors import hea
from qadence.execution import run
from qadence.operations import RX, RY, RZ, H, HamEvo, X
from qadence.states import equivalent_state
from qadence.transpile import apply_fn_to_blocks, chain_single_qubit_ops, digitalize, flatten
from qadence.types import LTSOrder


def test_flatten() -> None:
    from qadence.transpile.flatten import _flat_blocks

    x: AbstractBlock

    # make sure we get the identity when flattening non-existent blocks
    x = kron(X(0), X(1), X(2))
    assert tuple(_flat_blocks(x, ChainBlock)) == (X(0), X(1), X(2))
    assert flatten(x, [ChainBlock]) == kron(X(0), X(1), X(2))

    x = chain(chain(chain(chain(X(0)))))
    assert flatten(x, [ChainBlock]) == chain(X(0))

    x = kron(kron(X(0), kron(X(1))), kron(X(2)))
    assert tuple(_flat_blocks(x, KronBlock)) == (X(0), X(1), X(2))
    assert flatten(x, [KronBlock]) == kron(X(0), X(1), X(2))
    assert flatten(x, [KronBlock, ChainBlock]) == kron(X(0), X(1), X(2))

    x = chain(kron(X(0), kron(X(1))), kron(X(1)))
    assert flatten(x) == chain(kron(X(0), X(1)), kron(X(1)))

    x = 2 * kron(kron(X(0), kron(X(1))), kron(X(2)))
    assert flatten(x) == 2 * kron(X(0), X(1), X(2))

    x = kron(kron(X(0), 2 * kron(X(1))), kron(X(2)))
    # note that the innermost `KronBlock` behind the `ScaleBlock` stays
    assert flatten(x) == kron(X(0), 2 * kron(X(1)), X(2))

    x = chain(chain(chain(X(0))), kron(kron(X(0))))
    assert flatten(x, [ChainBlock]) == chain(X(0), kron(kron(X(0))))
    assert flatten(x, [KronBlock]) == chain(chain(chain(X(0))), kron(X(0)))
    assert flatten(x, [ChainBlock, KronBlock]) == chain(X(0), kron(X(0)))
    assert flatten(x, [AddBlock]) == x

    x = chain(kron(chain(chain(X(0), X(0)))))
    assert flatten(x, [ChainBlock]) == chain(kron(chain(X(0), X(0))))


def test_digitalize() -> None:
    x = chain(chain(X(0), HamEvo(X(0), 2), RX(0, 2)))
    assert str(digitalize(x, LTSOrder.BASIC)) == str(
        chain(chain(X(0), chain(H(0), RZ(0, 4.0), H(0)), RX(0, 2.0)))
    )


def test_chain_of_krons() -> None:
    b = chain(kron(X(0), X(2)), kron(X(0), X(2)))
    assert chain_single_qubit_ops(b) == kron(chain(X(0), X(0)), chain(X(2), X(2)))


def test_apply_fn() -> None:
    n_qubits = 4
    depth = 4

    def rot_replace(block: AbstractBlock) -> AbstractBlock:
        if isinstance(block, RX):
            return RY(block.qubit_support[0], block.parameters)
        return block

    block = hea(n_qubits, depth=depth, operations=[RX, RZ, RX])

    block_replace = apply_fn_to_blocks(block, rot_replace)

    block_target = hea(n_qubits, depth=depth, operations=[RY, RZ, RY])

    assert equivalent_state(run(block_replace), run(block_target))
