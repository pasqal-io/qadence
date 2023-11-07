from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
from numpy.random import rand
from sympy import acos

import qadence as qd
from qadence import BackendName
from qadence.blocks import (
    AbstractBlock,
    add,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.divergences import js_divergence
from qadence.measurements.protocols import Measurements
from qadence.models import QuantumModel
from qadence.noise import Noise
from qadence.noise.readout import WhiteNoise, bs_corruption, create_noise_matrix, sample_to_matrix
from qadence.operations import (
    CNOT,
    RX,
    RZ,
    H,
    HamEvo,
    X,
    Y,
    Z,
)
from qadence.types import DiffMode


@pytest.mark.parametrize(
    "error_probability, counters, exp_corrupted_counters, n_qubits",
    [
        (
            1.0,
            [Counter({"00": 27, "01": 23, "10": 24, "11": 26})],
            [Counter({"11": 27, "10": 23, "01": 24, "00": 26})],
            2,
        ),
        (
            1.0,
            [Counter({"001": 27, "010": 23, "101": 24, "110": 26})],
            [Counter({"110": 27, "101": 23, "010": 24, "001": 26})],
            3,
        ),
    ],
)
def test_bitstring_corruption_all_bitflips(
    error_probability: float, counters: list, exp_corrupted_counters: list, n_qubits: int
) -> None:
    n_shots = 100
    noise_matrix = create_noise_matrix(WhiteNoise.UNIFORM, n_shots, n_qubits)
    err_idx = np.array([(item).numpy() for i, item in enumerate(noise_matrix < error_probability)])
    sample = sample_to_matrix(counters[0])
    corrupted_counters = [
        bs_corruption(n_shots=n_shots, err_idx=err_idx, sample=sample, n_qubits=n_qubits)
    ]
    assert sum(corrupted_counters[0].values()) == n_shots
    assert corrupted_counters == exp_corrupted_counters
    assert torch.allclose(
        torch.tensor(1.0 - js_divergence(corrupted_counters[0], counters[0])),
        torch.ones(1),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "error_probability, counters, n_qubits",
    [
        (
            rand(),
            [Counter({"00": 27, "01": 23, "10": 24, "11": 26})],
            2,
        ),
        (
            rand(),
            [Counter({"001": 27, "010": 23, "101": 24, "110": 26})],
            3,
        ),
    ],
)
def test_bitstring_corruption_mixed_bitflips(
    error_probability: float, counters: list, n_qubits: int
) -> None:
    n_shots = 100
    noise_matrix = create_noise_matrix(WhiteNoise.UNIFORM, n_shots, n_qubits)
    err_idx = np.array([(item).numpy() for i, item in enumerate(noise_matrix < error_probability)])
    sample = sample_to_matrix(counters[0])
    corrupted_counters = [
        bs_corruption(n_shots=n_shots, err_idx=err_idx, sample=sample, n_qubits=n_qubits)
    ]
    for noiseless, noisy in zip(counters, corrupted_counters):
        assert sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) > 0.0


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, 1000, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.15, 1000, add(Z(0), Z(1), Z(2)), BackendName.BRAKET),
        (0.1, 5000, kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), BackendName.BRAKET),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, 2000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
        (0.1, 1300, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.BRAKET),
        (
            0.05,
            1500,
            kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
            + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
            BackendName.PULSER,
        ),
        (0.001, 5000, HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05), BackendName.BRAKET),
        (0.12, 2000, HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001), BackendName.BRAKET),
        (
            0.1,
            1000,
            HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
            BackendName.BRAKET,
        ),
        (0.1, 100, kron(X(0), X(1)), BackendName.PYQTORCH),
        (0.1, 200, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.PYQTORCH),
        (0.01, 1000, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH),
        (
            0.1,
            2000,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
            BackendName.PYQTORCH,
        ),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.PYQTORCH),
        (0.05, 10000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.PYQTORCH),
        (0.2, 1000, hamiltonian_factory(4, detuning=Z), BackendName.PYQTORCH),
        (0.1, 500, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.PYQTORCH),
    ],
)
def test_readout_error_quantum_model(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"

    noiseless_samples: list[Counter] = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample(n_shots=n_shots)

    noisy_samples: list[Counter] = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample(noise=Noise(protocol=Noise.READOUT), n_shots=n_shots)

    for noiseless, noisy in zip(noiseless_samples, noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(noiseless, noisy)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )


@pytest.mark.parametrize("backend", [BackendName.BRAKET, BackendName.PYQTORCH, BackendName.PULSER])
def test_readout_error_backends(backend: BackendName) -> None:
    n_qubits = 5
    error_probability = 0.1
    fp = qd.FeatureParameter("phi")
    feature_map = qd.kron(RX(i, 2 * acos(fp)) for i in range(n_qubits))
    inputs = {"phi": torch.rand(1)}
    # sample
    samples = qd.sample(feature_map, n_shots=1000, values=inputs, backend=backend, noise=None)
    # introduce noise
    options = {"error_probability": error_probability}
    noise = Noise(protocol=Noise.READOUT, options=options).get_noise_fn()
    noisy_samples = noise(counters=samples, n_qubits=n_qubits)
    # compare that the results are with an error of 10% (the default error_probability)
    for sample, noisy_sample in zip(samples, noisy_samples):
        assert sum(sample.values()) == sum(noisy_sample.values())
        assert js_divergence(sample, noisy_sample) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(sample, noisy_sample)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )


# TODO: Use strategies to test against randomly generated circuits.
@pytest.mark.parametrize(
    "measurement_proto, options",
    [
        (Measurements.TOMOGRAPHY, {"n_shots": 10000}),
        (Measurements.SHADOW, {"accuracy": 0.1, "confidence": 0.1}),
    ],
)
def test_readout_error_with_measurements(
    measurement_proto: Measurements,
    options: dict,
) -> None:
    circuit = QuantumCircuit(2, kron(H(0), Z(1)))
    inputs: dict = dict()
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

    model = QuantumModel(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    noise = Noise(protocol=Noise.READOUT)
    measurement = Measurements(protocol=str(measurement_proto), options=options)

    noisy = model.expectation(values=inputs, measurement=measurement, noise=noise)
    exact = model.expectation(values=inputs)
    if exact.numel() > 1:
        for noisy_value, exact_value in zip(noisy, exact):
            exact_val = torch.abs(exact_value).item()
            atol = exact_val / 3.0 if exact_val != 0.0 else 0.33
            assert torch.allclose(noisy_value, exact_value, atol=atol)
    else:
        exact_value = torch.abs(exact).item()
        atol = exact_value / 3.0 if exact_value != 0.0 else 0.33
        assert torch.allclose(noisy, exact, atol=atol)
