from __future__ import annotations

import pytest
from braket.circuits.gates import (
    CNot,
    CPhaseShift,
    Rx,
    Ry,
    Rz,
    Swap,
)
from braket.circuits.gates import (
    H as Braket_H,
)
from braket.circuits.gates import (
    S as Braket_S,
)
from braket.circuits.gates import (
    T as Braket_T,
)
from braket.circuits.gates import (
    X as Braket_X,
)
from braket.circuits.gates import (
    Y as Braket_Y,
)
from braket.circuits.gates import (
    Z as Braket_Z,
)
from braket.circuits.instruction import Instruction

from qadence.backends.braket.convert_ops import convert_block
from qadence.blocks import AbstractBlock
from qadence.operations import CNOT, CPHASE, RX, RY, RZ, SWAP, H, S, T, X, Y, Z


@pytest.mark.parametrize(
    "Qadence_op, braket_op",
    [
        (CNOT(0, 1), CNot.cnot(0, 1)),
        (CPHASE(0, 1, 0.5), CPhaseShift.cphaseshift(0, 1, 0.5)),
        (H(0), Braket_H.h(0)),
        (S(0), Braket_S.s(0)),
        (SWAP(0, 1), Swap.swap(0, 1)),
        (T(0), Braket_T.t(0)),
        (X(0), Braket_X.x(0)),
        (Y(0), Braket_Y.y(0)),
        (Z(0), Braket_Z.z(0)),
        (RX(0, 0.5), Rx.rx(0, 0.5)),
        (RY(0, 0.5), Ry.ry(0, 0.5)),
        (RZ(0, 0.5), Rz.rz(0, 0.5)),
    ],
)
def test_block_conversion(Qadence_op: AbstractBlock, braket_op: Instruction) -> None:
    assert convert_block(Qadence_op)[0] == braket_op
