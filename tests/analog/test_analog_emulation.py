from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from metrics import JS_ACCEPTANCE

from qadence.blocks import AbstractBlock, AnalogBlock, chain, kron
from qadence.execution import sample
from qadence.operations import (
    AnalogInteraction,
    AnalogRot,
    AnalogRX,
    AnalogRY,
)
from qadence.overlap import js_divergence
from qadence.register import Register
from qadence.types import PI


def layer(Op: Any, n_qubits: int, angle: float) -> AbstractBlock:
    return kron(Op(i, angle) for i in range(n_qubits))


d = 3.75


# @pytest.mark.parametrize(
#     "analog, digital_fn",
#     [
#         # FIXME: I commented this test because it was still running
#         # and failing despite the pytest.mark.xfail.
#         # pytest.param(  # enable with next pulser release
#         #     AnalogInteraction(duration=1), lambda n: I(n), marks=pytest.mark.xfail
#         # ),
#         (AnalogRX(angle=PI), lambda n: layer(RX, n, PI)),
#         (AnalogRY(angle=PI), lambda n: layer(RY, n, PI)),
#         (AnalogRZ(angle=PI), lambda n: layer(RZ, n, PI)),
#     ],
# )
# @pytest.mark.parametrize(
#     "register",
#     [
#         Register.from_coordinates([(0, 0)]),
#         Register.from_coordinates([(-d, 0), (d, 0)]),
#         Register.from_coordinates([(-d, 0), (d, 0), (0, d)]),
#         Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d)]),
#         Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d), (0, 0)]),
#         Register.from_coordinates([(-d, 0), (d, 0), (0, d), (0, -d), (0, 0), (d, d)]),
#     ],
# )
# def test_far_add_interaction(
# analog: AnalogBlock, digital_fn: Callable, register: Register) -> None:
#     register = register.rescale_coords(scaling=8.0)
#     emu_samples = sample(register, analog, backend="pyqtorch")[0]  # type: ignore[arg-type]
#     pulser_samples = sample(register, analog, backend="pulser")[0]  # type: ignore[arg-type]
#     assert js_divergence(pulser_samples, emu_samples) < JS_ACCEPTANCE

#     wf = random_state(register.n_qubits)
#     digital = digital_fn(register.n_qubits)
#     emu_state = run(register, analog, state=wf)
#     dig_state = run(register, digital, state=wf)
#     assert equivalent_state(emu_state, dig_state, atol=1e-3)


@pytest.mark.parametrize(
    "block",
    [
        AnalogRX((0, 1, 2, 3), angle=PI),
        AnalogRY((0, 1, 2, 3), angle=PI),
        chain(AnalogInteraction(duration=2000), AnalogRX((0, 1, 2, 3), angle=PI)),
        chain(
            AnalogRot((0, 1, 2, 3), duration=1000, omega=1.0, delta=0.0, phase=0),
            AnalogRot((0, 1, 2, 3), duration=1000, omega=0.0, delta=1.0, phase=0),
        ),
        # kron(AnalogRX(PI, qubit_support=(0, 1)), AnalogInteraction(1000, qubit_support=(2, 3))),
    ],
)
@pytest.mark.parametrize("register", [Register.from_coordinates([(0, 5), (5, 5), (5, 0), (0, 0)])])
@pytest.mark.flaky(max_runs=5)
def test_close_add_interaction(block: AnalogBlock, register: Register) -> None:
    register = register.rescale_coords(scaling=8.0)
    pulser_samples = sample(register, block, backend="pulser", n_shots=1000)[0]
    pyqtorch_samples = sample(register, block, backend="pyqtorch", n_shots=1000)[0]
    assert js_divergence(pulser_samples, pyqtorch_samples) < JS_ACCEPTANCE


def test_mixing_digital_analog() -> None:
    from qadence import X, chain, kron

    b = chain(kron(X(0), X(1)), AnalogRX((0, 1), PI))
    r = Register.from_coordinates([(0, 10), (0, -10)])

    sample_results = sample(r, b)[0]

    assert js_divergence(sample_results, Counter({"00": 100})) < JS_ACCEPTANCE


# FIXME: Adapt when custom interaction functions are again supported
# def test_custom_interaction_function() -> None:
#     circuit = QuantumCircuit(2, AnalogInteraction(duration=100))
#     emulated = add_interaction(circuit, interaction=lambda reg, pairs: I(0))
#     assert emulated.block == HamEvo(I(0), 100 / 1000)

#     m = QuantumModel(circuit, configuration={"interaction": lambda reg, pairs: I(0)})
#     assert m._circuit.abstract.block == HamEvo(I(0), 100 / 1000)
