from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import pytest
from metrics import JS_ACCEPTANCE
from torch import pi

from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import AnalogBlock
from qadence.execution import run, sample
from qadence.operations import (
    RX,
    RY,
    RZ,
    AnalogRot,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    chain,
    kron,
    wait,
)
from qadence.overlap import js_divergence
from qadence.register import Register
from qadence.states import equivalent_state, random_state


def layer(Op: Any, n_qubits: int, angle: float) -> AbstractBlock:
    return kron(Op(i, angle) for i in range(n_qubits))


d = 3.75


@pytest.mark.parametrize(
    "analog, digital_fn",
    [
        # FIXME: I commented this test because it was still running
        # and failing despite the pytest.mark.xfail.
        # pytest.param(  # enable with next pulser release
        #     wait(duration=1), lambda n: I(n), marks=pytest.mark.xfail
        # ),
        (AnalogRX(angle=pi), lambda n: layer(RX, n, pi)),
        (AnalogRY(angle=pi), lambda n: layer(RY, n, pi)),
        (AnalogRZ(angle=pi), lambda n: layer(RZ, n, pi)),
    ],
)
@pytest.mark.parametrize(
    "register",
    [
        Register.from_coordinates([(0, 0)]),
        Register.from_coordinates([(-d, 0), (d, 0)]),
        Register.from_coordinates([(-d, 0), (d, 0), (0, d)]),
        Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d)]),
        Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d), (0, 0)]),
        Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d), (0, 0), (d, d)]),
    ],
)
def test_far_add_interaction(analog: AnalogBlock, digital_fn: Callable, register: Register) -> None:
    config = {"spacing": 8.0}
    emu_samples = sample(register, analog, backend="pyqtorch", configuration=config)[0]
    pulser_samples = sample(register, analog, backend="pulser", configuration=config)[0]
    assert js_divergence(pulser_samples, emu_samples) < JS_ACCEPTANCE

    wf = random_state(register.n_qubits)
    digital = digital_fn(register.n_qubits)
    emu_state = run(register, analog, state=wf, configuration=config)
    dig_state = run(register, digital, state=wf, configuration=config)
    assert equivalent_state(emu_state, dig_state, atol=1e-3)


@pytest.mark.parametrize(
    "block",
    [
        AnalogRX(angle=pi),
        AnalogRY(angle=pi),
        chain(wait(duration=2000), AnalogRX(angle=pi)),
        chain(
            AnalogRot(duration=1000, omega=1.0, delta=0.0, phase=0),
            AnalogRot(duration=1000, omega=0.0, delta=1.0, phase=0),
        ),
        # kron(AnalogRX(pi, qubit_support=(0, 1)), wait(1000, qubit_support=(2, 3))),
    ],
)
@pytest.mark.parametrize("register", [Register.from_coordinates([(0, 5), (5, 5), (5, 0), (0, 0)])])
@pytest.mark.flaky(max_runs=5)
def test_close_add_interaction(block: AnalogBlock, register: Register) -> None:
    config = {"spacing": 8.0}
    pulser_samples = sample(register, block, backend="pulser", n_shots=1000, configuration=config)[
        0
    ]
    pyqtorch_samples = sample(
        register, block, backend="pyqtorch", n_shots=1000, configuration=config
    )[0]
    assert js_divergence(pulser_samples, pyqtorch_samples) < JS_ACCEPTANCE


def test_mixing_digital_analog() -> None:
    from qadence import X, chain, kron

    b = chain(kron(X(0), X(1)), AnalogRX(pi))
    r = Register.from_coordinates([(0, 10), (0, -10)])

    assert js_divergence(sample(r, b)[0], Counter({"00": 100})) < JS_ACCEPTANCE


# FIXME: Adapt when custom interaction functions are again supported
# def test_custom_interaction_function() -> None:
#     circuit = QuantumCircuit(2, wait(duration=100))
#     emulated = add_interaction(circuit, interaction=lambda reg, pairs: I(0))
#     assert emulated.block == HamEvo(I(0), 100 / 1000)

#     m = QuantumModel(circuit, configuration={"interaction": lambda reg, pairs: I(0)})
#     assert m._circuit.abstract.block == HamEvo(I(0), 100 / 1000)
