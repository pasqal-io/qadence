from __future__ import annotations

from qadence import QuantumCircuit, chain, hea, kron
from qadence.blocks import AbstractBlock, AddBlock, ChainBlock, KronBlock
from qadence.operations import RX, RZ, H, HamEvo, I, S, X, Y, Z
from qadence.transpile import digitalize, flatten, reverse
from qadence.types import LTSOrder


def test_flatten() -> None:
    from qadence.transpile.block import _flat_blocks

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
    assert digitalize(x, LTSOrder.BASIC) == chain(
        chain(X(0), chain(H(0), RZ(0, 4.0), H(0)), RX(0, 2.0))
    )


def test_reverse() -> None:
    x = chain(X(0), Y(0), Z(0))
    qc = QuantumCircuit(1, x)
    rev_expected = chain(Z(0), Y(0), X(0))
    assert reverse(qc) == QuantumCircuit(1, rev_expected)
    assert reverse(qc.block) == rev_expected

    x = chain(chain(X(0), Y(0), Z(0)), chain(I(0), H(0), S(0)))
    assert chain(chain(S(0), H(0), I(0)), chain(Z(0), Y(0), X(0))) == reverse(x)

    x = hea(2, 2)  # type: ignore[assignment]
    assert reverse(reverse(x)) == x

    x = chain(2 * RX(0, "theta"))
    assert reverse(reverse(x)) == x
