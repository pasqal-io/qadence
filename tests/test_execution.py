from __future__ import annotations

from collections import Counter

import pytest
import strategies as st  # type: ignore
from hypothesis import given, settings
from torch import Tensor, rand

from qadence import RX, QuantumCircuit, Z, expectation, run, sample, total_magnetization
from qadence.backend import BackendName
from qadence.blocks import AbstractBlock
from qadence.register import Register

BACKENDS = [BackendName.PYQTORCH, BackendName.BRAKET]


@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_run(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    circ, inputs = circ_and_vals
    for backend in BACKENDS:
        reg = Register(circ.n_qubits)
        wf = run(circ, values=inputs, backend=backend)  # type: ignore[arg-type]
        wf = run(reg, circ.block, values=inputs, backend=backend)  # type: ignore[arg-type]
        wf = run(circ.block, values=inputs, backend=backend)  # type: ignore[arg-type]
        assert isinstance(wf, Tensor)


@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_sample(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    circ, inputs = circ_and_vals
    reg = Register(circ.n_qubits)
    for backend in BACKENDS:
        samples = sample(circ, values=inputs, backend=backend)
        samples = sample(reg, circ.block, values=inputs, backend=backend)
        samples = sample(circ.block, values=inputs, backend=backend)
        assert all([isinstance(s, Counter) for s in samples])


@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_expectation(circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]) -> None:
    circ, inputs = circ_and_vals
    reg = Register(circ.n_qubits)
    obs = total_magnetization(reg.n_qubits)
    for backend in BACKENDS:
        x = expectation(circ, obs, values=inputs, backend=backend)  # type: ignore[call-arg]
        x = expectation(reg, circ.block, obs, values=inputs, backend=backend)  # type: ignore
        x = expectation(circ.block, obs, values=inputs, backend=backend)  # type: ignore[call-arg]
        if inputs:
            assert x.size(0) == len(inputs[list(inputs.keys())[0]])
        else:
            assert x.size(0) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_qubit_block(
    backend: BackendName, block: AbstractBlock = RX(2, rand(1).item())
) -> None:
    run(block, values={}, backend=backend)  # type: ignore[arg-type]
    sample(block, values={}, backend=backend)  # type: ignore[arg-type]
    expectation(block, Z(0), values={}, backend=backend)  # type: ignore[arg-type]
