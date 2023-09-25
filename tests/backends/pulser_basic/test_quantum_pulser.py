from __future__ import annotations

from collections import Counter

import pytest
import torch
from metrics import JS_ACCEPTANCE

from qadence import (
    RX,
    RY,
    AnalogRot,
    BackendName,
    FeatureParameter,
    QuantumCircuit,
    Register,
    VariationalParameter,
    backend_factory,
    chain,
    entangle,
    kron,
    total_magnetization,
)
from qadence.backends.pulser import Device
from qadence.divergences import js_divergence


@pytest.fixture
def batched_circuit() -> QuantumCircuit:
    n_qubits = 3
    phi = FeatureParameter("phi")
    theta = VariationalParameter("theta")

    block = kron(RX(0, phi), RX(1, theta), RX(2, torch.pi))
    return QuantumCircuit(n_qubits, block)


@pytest.mark.parametrize(
    "circuit,goal",
    [
        (
            QuantumCircuit(
                Register(2), chain(entangle(383, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2))
            ),
            Counter({"00": 250, "11": 250}),
        ),
        (
            QuantumCircuit(
                Register.square(qubits_side=2),
                chain(
                    entangle(2488),
                    AnalogRot(duration=300, omega=5 * torch.pi, delta=0, phase=0),
                ),
            ),
            Counter(
                {
                    "1111": 145,
                    "1110": 15,
                    "1101": 15,
                    "1100": 15,
                    "1011": 15,
                    "1010": 15,
                    "1001": 15,
                    "1000": 15,
                    "0111": 15,
                    "0110": 15,
                    "0101": 15,
                    "0100": 15,
                    "0011": 15,
                    "0010": 15,
                    "0001": 15,
                    "0000": 145,
                }
            ),
        ),
    ],
)
def test_pulser_sequence_sample(circuit: QuantumCircuit, goal: Counter) -> None:
    config = {"device_type": Device.REALISTIC}
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None, configuration=config)
    sample = backend.sample(backend.circuit(circuit), {}, n_shots=500)[0]
    assert js_divergence(sample, goal) < JS_ACCEPTANCE


def test_expectation_batched(batched_circuit: QuantumCircuit) -> None:
    batch_size = 3
    values = {"phi": torch.tensor([torch.pi / 5, torch.pi / 4, torch.pi / 3])}
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
    values = {"phi": torch.tensor([torch.pi / 5, torch.pi / 4, torch.pi / 3])}

    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    circ, _, embed, params = backend.convert(batched_circuit)
    wf = backend.run(circ, param_values=embed(params, values))

    assert wf.shape == (batch_size, 2**batched_circuit.n_qubits)
