from __future__ import annotations

import pytest
from torch import pi, tensor

from qadence import (
    RX,
    BackendName,
    FeatureParameter,
    QuantumCircuit,
    VariationalParameter,
    backend_factory,
    kron,
    total_magnetization,
)


@pytest.fixture
def batched_circuit() -> QuantumCircuit:
    n_qubits = 3
    phi = FeatureParameter("phi")
    theta = VariationalParameter("theta")

    block = kron(RX(0, phi), RX(1, theta), RX(2, pi))
    return QuantumCircuit(n_qubits, block)


def test_expectation_batched(batched_circuit: QuantumCircuit) -> None:
    batch_size = 3
    values = {"phi": tensor([pi / 5, pi / 4, pi / 3])}
    observables = [
        total_magnetization(batched_circuit.n_qubits),
        2 * total_magnetization(batched_circuit.n_qubits),
    ]

    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    circ, obs, embed, params = backend.convert(batched_circuit, observable=observables)
    expval = backend.expectation(circ, observable=obs, param_values=embed(params, values))
    assert expval.shape == (batch_size, len(observables))


def test_run_batched(batched_circuit: QuantumCircuit) -> None:
    batch_size = 3
    values = {"phi": tensor([pi / 5, pi / 4, pi / 3])}

    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    circ, _, embed, params = backend.convert(batched_circuit)
    wf = backend.run(circ, param_values=embed(params, values))

    assert wf.shape == (batch_size, 2**batched_circuit.n_qubits)
