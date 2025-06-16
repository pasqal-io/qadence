from __future__ import annotations

from collections import Counter

import pytest
import torch
from sympy import acos

import qadence as qd
from qadence import BackendName, QuantumModel, DiffMode
from qadence.blocks import (
    AbstractBlock,
    add,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.divergences import js_divergence
from qadence.noise import NoiseHandler
from qadence.operations import (
    CNOT,
    RX,
    HamEvo,
    X,
    Y,
    Z,
)
from qadence.noise import available_protocols


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, block",
    [
        (
            0.1,
            100,
            kron(X(0), X(1)),
        ),
        (
            0.1,
            200,
            kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
        ),
        (
            0.01,
            1000,
            add(Z(0), Z(1), Z(2)),
        ),
        (
            0.1,
            2000,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
        ),
        (
            0.1,
            500,
            add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
        ),
        (0.05, 10000, add(kron(Z(0), Z(1)), kron(X(2), X(3)))),
        (0.2, 1000, hamiltonian_factory(4, detuning=Z)),
        (0.1, 500, kron(Z(0), Z(1)) + CNOT(0, 1)),
    ],
)
def test_readout_error_quantum_model(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
) -> None:
    backend = BackendName.PYQTORCH
    diff_mode = DiffMode.AD
    model = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    )
    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)

    noise_protocol = available_protocols.IndependentReadout(error_definition=0.1)
    noisy_samples: list[Counter] = model.sample(noise=noise_protocol, n_shots=n_shots)

    for noiseless, noisy in zip(noiseless_samples, noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) >= 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(noiseless, noisy)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )

    rand_confusion = torch.rand(2**block.n_qubits, 2**block.n_qubits)
    rand_confusion = rand_confusion / rand_confusion.sum(dim=1, keepdim=True)
    corr_noise_protocol = available_protocols.CorrelatedReadout(error_definition=rand_confusion)
    # assert difference with noiseless samples
    corr_noisy_samples: list[Counter] = model.sample(noise=corr_noise_protocol, n_shots=n_shots)
    for noiseless, noisy in zip(noiseless_samples, corr_noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) >= 0.0

    # assert difference noisy samples
    for noisy, corr_noisy in zip(noisy_samples, corr_noisy_samples):
        assert sum(noisy.values()) == sum(corr_noisy.values()) == n_shots
        assert js_divergence(noisy, corr_noisy) >= 0.0


@pytest.mark.parametrize("backend", [BackendName.PYQTORCH, BackendName.PULSER])
def test_readout_error_backends(backend: BackendName) -> None:
    n_qubits = 5
    error_probability = 0.1
    fp = qd.FeatureParameter("phi")
    feature_map = qd.kron(RX(i, 2 * acos(fp)) for i in range(n_qubits))
    inputs = {"phi": torch.rand(1)}
    # sample
    samples = qd.sample(feature_map, n_shots=1000, values=inputs, backend=backend, noise=None)
    # introduce noise
    noise = available_protocols.IndependentReadout(error_definition=error_probability)
    noisy_samples = qd.sample(
        feature_map, n_shots=1000, values=inputs, backend=backend, noise=noise
    )
    # compare that the results are with an error of 10% (the default error_probability)
    for sample, noisy_sample in zip(samples, noisy_samples):
        assert sum(sample.values()) == sum(noisy_sample.values())
        assert js_divergence(sample, noisy_sample) >= 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(sample, noisy_sample)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )


# # TODO: Use strategies to test against randomly generated circuits.
# TODO: re-enable with a new release of pyqtorch after 1.7.0
# @pytest.mark.parametrize(
#     "measurement_proto, options",
#     [
#         (Measurements.TOMOGRAPHY, {"n_shots": 10000}),
#         (Measurements.SHADOW, {"accuracy": 0.1, "confidence": 0.1}),
#     ],
# )
# def test_readout_error_with_measurements(
#     measurement_proto: Measurements,
#     options: dict,
# ) -> None:
#     circuit = QuantumCircuit(2, kron(H(0), Z(1)))
#     inputs: dict = dict()
#     observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

#     model = QuantumModel(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
#     noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT)
#     measurement = Measurements(protocol=str(measurement_proto), options=options)

#     noisy = model.expectation(values=inputs, measurement=measurement, noise=noise)
#     exact = model.expectation(values=inputs)
#     if exact.numel() > 1:
#         for noisy_value, exact_value in zip(noisy, exact):
#             exact_val = torch.abs(exact_value).item()
#             atol = exact_val / 3.0 if exact_val != 0.0 else 0.33
#             assert torch.allclose(noisy_value, exact_value, atol=atol)
#     else:
#         exact_value = torch.abs(exact).item()
#         atol = exact_value / 3.0 if exact_value != 0.0 else 0.33
#         assert torch.allclose(noisy, exact, atol=atol)


def test_serialization() -> None:
    noise = available_protocols.IndependentReadout(error_definition=0.1)
    serialized_noise = available_protocols.IndependentReadout(noise.model_dump())
    assert noise == serialized_noise

    rand_confusion = torch.rand(4, 4)
    rand_confusion = rand_confusion / rand_confusion.sum(dim=1, keepdim=True)
    noise = available_protocols.CorrelatedReadout(seed=0, confusion_matrix=rand_confusion)
    serialized_noise = available_protocols.CorrelatedReadout(noise.model_dump())
    assert noise == serialized_noise