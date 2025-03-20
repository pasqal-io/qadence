from __future__ import annotations

import pytest

from qadence.blocks import CompositeBlock
from qadence.blocks.analog import (
    AnalogBlock,
    ConstantAnalogRotation,
    QubitSupport,
    chain,
    kron,
)
from qadence.operations import AnalogInteraction, AnalogRX, X
from qadence.parameters import ParamMap
from qadence.types import PI


def test_qubit_support() -> None:
    assert QubitSupport("global").is_global
    assert not QubitSupport(2, 3).is_global

    assert QubitSupport("global") + QubitSupport("global") == QubitSupport("global")
    assert QubitSupport("global") + QubitSupport(1, 2) == QubitSupport(0, 1, 2)
    assert QubitSupport(2, 3) + QubitSupport(1, 2) == QubitSupport(1, 2, 3)

    # local QubitSupport / mixing QubitSupport & tuple
    assert QubitSupport(0, 1) + (2, 4) == QubitSupport(0, 1, 2, 4)
    assert (0, 4) + QubitSupport(1, 2) == QubitSupport(0, 1, 2, 4)
    assert (0, 4) + QubitSupport("global") == QubitSupport(0, 1, 2, 3, 4)
    assert QubitSupport("global") + (0, 4) == QubitSupport(0, 1, 2, 3, 4)
    assert QubitSupport("global") + () == QubitSupport("global")
    assert () + QubitSupport("global") == QubitSupport("global")
    assert () + QubitSupport(1, 2) == QubitSupport(1, 2)
    assert QubitSupport() == ()


def test_analog_block() -> None:
    b: AnalogBlock
    b = AnalogInteraction(duration=3, qubit_support=(1, 2))
    assert repr(b) == "InteractionBlock(t=3, support=(1, 2))"

    c1 = chain(
        ConstantAnalogRotation(parameters=ParamMap(duration=2000, omega=1, delta=0, phase=0)),
        ConstantAnalogRotation(parameters=ParamMap(duration=3000, omega=1, delta=0, phase=0)),
    )
    assert c1.duration.equals(5000)
    assert c1.qubit_support == QubitSupport("global")

    c2 = kron(
        AnalogRX(PI, qubit_support=(0, 1)),
        AnalogInteraction(duration=1000, qubit_support=(2, 3)),
    )
    assert c2.duration.equals(1000)
    assert c2.qubit_support == QubitSupport(0, 1, 2, 3)

    c3 = chain(
        kron(
            AnalogRX(PI, qubit_support=(0, 1)),
            AnalogInteraction(duration=1000, qubit_support=(2, 3)),
        ),
        kron(
            AnalogInteraction(duration=1000, qubit_support=(0, 1)),
            AnalogRX(PI, qubit_support=(2, 3)),
        ),
    )
    assert c3.duration.equals(2000)

    with pytest.raises(ValueError, match="Only KronBlocks or global blocks can be chain'ed."):
        chain(c3, AnalogInteraction(duration=10))

    with pytest.raises(ValueError, match="Blocks with global support cannot be kron'ed."):
        kron(AnalogRX(PI, qubit_support=(0, 1)), AnalogInteraction(duration=1000))

    with pytest.raises(ValueError, match="Make sure blocks act on distinct qubits!"):
        kron(
            AnalogRX(PI, qubit_support=(0, 1)),
            AnalogInteraction(duration=1000, qubit_support=(1, 2)),
        )

    with pytest.raises(ValueError, match="Kron'ed blocks have to have same duration."):
        kron(
            AnalogRX(1, qubit_support=(0, 1)),
            AnalogInteraction(duration=10, qubit_support=(2, 3)),
        )


@pytest.mark.xfail
def test_mix_digital_analog() -> None:
    from qadence import chain

    b = chain(X(0), AnalogRX(2.0))
    assert b.qubit_support == (0,)

    b = chain(X(0), AnalogInteraction(2.0), X(2))
    assert b.qubit_support == (0, 1, 2)

    b = chain(chain(X(0), AnalogInteraction(2.0, qubit_support="global"), X(2)), X(3))
    assert all([not isinstance(b, CompositeBlock) for b in b.blocks])
    assert b.qubit_support == (0, 1, 2, 3)
