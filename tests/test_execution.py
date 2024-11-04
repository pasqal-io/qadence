from __future__ import annotations

from collections import Counter

import pytest
import strategies as st  # type: ignore
from hypothesis import given, settings
from metrics import JS_ACCEPTANCE  # type: ignore
from torch import Tensor, allclose, rand

from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.constructors import total_magnetization
from qadence.divergences import js_divergence
from qadence.execution import expectation, run, sample
from qadence.operations import RX, Z
from qadence.register import Register
from qadence.states import equivalent_state
from qadence.types import BackendName, DiffMode


@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_run(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    backend = BackendName.PYQTORCH
    circ, inputs = circ_and_vals
    reg = Register(circ.n_qubits)
    wf = run(circ, values=inputs, backend=backend)  # type: ignore[arg-type]
    wf = run(reg, circ.block, values=inputs, backend=backend)  # type: ignore[arg-type]
    wf = run(circ.block, values=inputs, backend=backend)  # type: ignore[arg-type]
    assert isinstance(wf, Tensor)


@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_sample(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    backend = BackendName.PYQTORCH
    circ, inputs = circ_and_vals
    reg = Register(circ.n_qubits)
    samples = sample(circ, values=inputs, backend=backend)
    samples = sample(reg, circ.block, values=inputs, backend=backend)
    samples = sample(circ.block, values=inputs, backend=backend)
    assert all([isinstance(s, Counter) for s in samples])


@pytest.mark.parametrize("diff_mode", list(DiffMode) + [None])
@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_expectation(
    diff_mode: DiffMode,
    circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]],
) -> None:
    backend = BackendName.PYQTORCH
    if diff_mode in ("ad", "adjoint") and backend != "pyqtorch":
        pytest.skip(f"Backend {backend} doesnt support diff_mode={diff_mode}.")
    circ, inputs = circ_and_vals
    reg = Register(circ.n_qubits)
    obs = total_magnetization(reg.n_qubits)
    x = expectation(
        circ, obs, values=inputs, backend=backend, diff_mode=diff_mode
    )  # type: ignore[call-arg]
    x = expectation(
        reg, circ.block, obs, values=inputs, backend=backend, diff_mode=diff_mode  # type: ignore
    )
    x = expectation(
        circ.block, obs, values=inputs, backend=backend, diff_mode=diff_mode
    )  # type: ignore[call-arg]
    if inputs:
        assert x.size(0) == len(inputs[list(inputs.keys())[0]])
    else:
        assert x.size(0) == 1


def test_single_qubit_block(block: AbstractBlock = RX(2, rand(1).item())) -> None:
    backend = BackendName.PYQTORCH
    run(block, values={}, backend=backend)  # type: ignore[arg-type]
    sample(block, values={}, backend=backend)  # type: ignore[arg-type]
    expectation(block, Z(0), values={}, backend=backend)  # type: ignore[arg-type]


@pytest.mark.flaky(max_runs=5)
@given(st.batched_digital_circuits())
@settings(deadline=None)
def test_singlequbit_comp(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    circ, inputs = circ_and_vals
    wf_0 = run(circ, values=inputs)  # type: ignore[arg-type]
    samples_0 = sample(circ, values=inputs)  # type: ignore[arg-type]
    expectation_0 = expectation(circ, Z(0), values=inputs)  # type: ignore[arg-type]

    # diffmode = "ad" makes pyq compose single qubit ops if possible

    wf_1 = run(circ, values=inputs)  # type: ignore[arg-type]
    samples_1 = sample(circ, values=inputs)  # type: ignore[arg-type]
    expectation_1 = expectation(circ, Z(0), values=inputs, diff_mode="ad")  # type: ignore[arg-type]

    assert equivalent_state(wf_0, wf_1)
    assert allclose(expectation_0, expectation_1)

    for sample0, sample1 in zip(samples_0, samples_1):
        assert js_divergence(sample0, sample1) < JS_ACCEPTANCE
