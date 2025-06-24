from __future__ import annotations

import pytest
import torch
from torch import Tensor, tensor

from qadence import DiffMode, AbstractNoise, QuantumModel
from qadence.backends import backend_factory
from qadence.blocks import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import total_magnetization
from qadence.operations import RX, AnalogRX, AnalogRZ, Z
from qadence.parameters import FeatureParameter, VariationalParameter
from qadence.types import PI, BackendName
from qadence.noise import available_protocols


@pytest.fixture
def batched_circuit() -> QuantumCircuit:
    n_qubits = 3
    phi = FeatureParameter("phi")
    theta = VariationalParameter("theta")

    block = kron(RX(0, phi), RX(1, theta), RX(2, PI))
    return QuantumCircuit(n_qubits, block)


def test_expectation_batched(batched_circuit: QuantumCircuit) -> None:
    batch_size = 3
    values = {"phi": tensor([PI / 5, PI / 4, PI / 3])}
    observables = [
        total_magnetization(batched_circuit.n_qubits),
        2 * total_magnetization(batched_circuit.n_qubits),
    ]

    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    circ, obs, embed, params = backend.convert(batched_circuit, observable=observables)
    expval = backend.expectation(circ, observable=obs, param_values=embed(params, values))
    assert expval.shape == (batch_size, len(observables))

    # try separated values
    values_sep = {"circuit": values}
    expval_sep = backend.expectation(circ, observable=obs, param_values=embed(params, values_sep))
    assert torch.allclose(expval, expval_sep)


def test_run_batched(batched_circuit: QuantumCircuit) -> None:
    batch_size = 3
    values = {"phi": tensor([PI / 5, PI / 4, PI / 3])}

    backend = backend_factory(backend=BackendName.PULSER, diff_mode=None)
    circ, _, embed, params = backend.convert(batched_circuit)
    wf = backend.run(circ, param_values=embed(params, values))

    assert wf.shape == (batch_size, 2**batched_circuit.n_qubits)


def test_noisy_simulations(noiseless_pulser_sim: Tensor, noisy_pulser_sim: Tensor) -> None:
    analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
    observable = [Z(0) + Z(1)]
    circuit = QuantumCircuit(2, analog_block)
    model_noiseless = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    noiseless_expectation = model_noiseless.expectation()

    noise = available_protocols.AnalogDepolarizing(error_definition=0.1)
    model_noisy = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        noise=noise,
    )
    noisy_expectation = model_noisy.expectation()
    assert torch.allclose(noiseless_expectation, noiseless_pulser_sim, atol=1.0e-3)
    assert torch.allclose(noisy_expectation, noisy_pulser_sim, atol=1.0e-3)

    # test backend itself
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=DiffMode.GPSR)
    (pulser_circ, pulser_obs, embed, params) = backend.convert(circuit, observable)
    native_noisy_expectation = backend.expectation(
        pulser_circ, pulser_obs, embed(params, {}), noise=noise
    )
    assert torch.allclose(native_noisy_expectation, noisy_expectation, atol=1.0e-3)


def test_batched_noisy_simulations(
    noiseless_pulser_sim: Tensor, batched_noisy_pulser_sim: Tensor
) -> None:
    analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
    observable = [Z(0) + Z(1)]
    circuit = QuantumCircuit(2, analog_block)
    model_noiseless = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    noiseless_expectation = model_noiseless.expectation()

    noise = available_protocols.Dephasing(
        error_definition=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    )
    model_noisy = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        noise=noise,
    )
    batched_noisy_expectation = model_noisy.expectation()
    assert torch.allclose(noiseless_expectation, noiseless_pulser_sim, atol=1.0e-3)
    assert torch.allclose(batched_noisy_expectation, batched_noisy_pulser_sim, atol=1.0e-3)

    # test backend itself
    backend = backend_factory(backend=BackendName.PULSER, diff_mode=DiffMode.GPSR)
    (pulser_circ, pulser_obs, embed, params) = backend.convert(circuit, observable)
    batched_native_expectation = backend.expectation(
        pulser_circ, pulser_obs, embed(params, {}), noise=noise
    )
    assert torch.allclose(batched_native_expectation, batched_noisy_pulser_sim, atol=1.0e-3)
