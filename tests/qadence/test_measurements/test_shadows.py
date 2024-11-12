from __future__ import annotations

import json
import os

import pytest
import torch

from qadence.backends.api import backend_factory
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import add, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import ising_hamiltonian, total_magnetization
from qadence.execution import expectation
from qadence.measurements import Measurements
from qadence.measurements.shadow import (
    _max_observable_weight,
    estimations,
    number_of_samples,
)
from qadence.model import QuantumModel
from qadence.operations import RX, RY, H, I, X, Y, Z
from qadence.parameters import Parameter
from qadence.serialization import deserialize
from qadence.types import BackendName, DiffMode


@pytest.mark.parametrize(
    "observable, exp_weight",
    [
        (X(0), 1),
        (kron(*[X(0), Y(1), Z(2)]), 3),
        (add(*[X(0), Y(0), Z(0)]), 1),
        (kron(*[X(0), H(1), I(2), Z(3)]), 2),
        (total_magnetization(5), 1),
        (ising_hamiltonian(4), 2),
    ],
)
def test_weight(observable: AbstractBlock, exp_weight: int) -> None:
    qubit_weight = _max_observable_weight(observable)
    assert qubit_weight == exp_weight


@pytest.mark.parametrize(
    "observables, accuracy, confidence, exp_samples",
    [([total_magnetization(2)], 0.1, 0.1, (10200, 6))],
)
def test_number_of_samples(
    observables: list[AbstractBlock], accuracy: float, confidence: float, exp_samples: tuple
) -> None:
    N, K = number_of_samples(observables=observables, accuracy=accuracy, confidence=confidence)
    assert N == exp_samples[0]
    assert K == exp_samples[1]


theta = Parameter("theta")


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "circuit, observable, values",
    [
        (QuantumCircuit(2, kron(X(0), X(1))), X(0) @ X(1), {}),
        (QuantumCircuit(2, kron(X(0), X(1))), X(0) @ Y(1), {}),
        (QuantumCircuit(2, kron(X(0), X(1))), Y(0) @ X(1), {}),
        (QuantumCircuit(2, kron(X(0), X(1))), Y(0) @ Y(1), {}),
        (QuantumCircuit(2, kron(Z(0), H(1))), X(0) @ Z(1), {}),
        (
            QuantumCircuit(2, kron(RX(0, theta), X(1))),
            kron(Z(0), Z(1)),
            {"theta": torch.tensor([0.5, 1.0])},
        ),
        (QuantumCircuit(2, kron(X(0), Z(1))), ising_hamiltonian(2), {}),
    ],
)
def test_estimations_comparison_exact(
    circuit: QuantumCircuit, observable: AbstractBlock, values: dict
) -> None:
    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    (conv_circ, _, embed, params) = backend.convert(circuit=circuit, observable=observable)
    param_values = embed(params, values)

    estimated_exp = estimations(
        circuit=conv_circ.abstract,
        observables=[observable],
        param_values=param_values,
        shadow_size=5000,
    )
    exact_exp = expectation(circuit, observable, values=values)
    assert torch.allclose(estimated_exp, exact_exp, atol=0.2)


theta1 = Parameter("theta1", trainable=False)
theta2 = Parameter("theta2", trainable=False)
theta3 = Parameter("theta3", trainable=False)
theta4 = Parameter("theta4", trainable=False)


blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3), RY(1, theta4)),
)

values = {
    "theta1": torch.tensor([0.5]),
    "theta2": torch.tensor([1.5]),
    "theta3": torch.tensor([2.0]),
    "theta4": torch.tensor([2.5]),
}

values2 = {
    "theta1": torch.tensor([0.5, 1.0]),
    "theta2": torch.tensor([1.5, 2.0]),
    "theta3": torch.tensor([2.0, 2.5]),
    "theta4": torch.tensor([2.5, 3.0]),
}


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "circuit, values, diff_mode",
    [
        (QuantumCircuit(2, blocks), values, DiffMode.AD),
        (QuantumCircuit(2, blocks), values2, DiffMode.GPSR),
    ],
)
@pytest.mark.parametrize("observable", [Z(0) ^ 2, X(1)])
def test_estimations_comparison_tomo_forward_pass(
    circuit: QuantumCircuit,
    values: dict,
    diff_mode: DiffMode,
    observable: AbstractBlock,
) -> None:
    pyq_backend = backend_factory(BackendName.PYQTORCH, diff_mode=diff_mode)
    (conv_circ, conv_obs, embed, params) = pyq_backend.convert(circuit, observable)
    pyq_exp_exact = pyq_backend.expectation(conv_circ, conv_obs, embed(params, values))
    model = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    options = {"n_shots": 100000}
    estimated_exp_tomo = model.expectation(
        values=values,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=options),
    )
    new_options = {"accuracy": 0.1, "confidence": 0.1}
    estimated_exp_shadow = model.expectation(
        values=values,
        measurement=Measurements(protocol=Measurements.SHADOW, options=new_options),
    )  # N = 54400.
    assert torch.allclose(estimated_exp_tomo, pyq_exp_exact, atol=1.0e-2)
    assert torch.allclose(estimated_exp_shadow, pyq_exp_exact, atol=0.1)
    assert torch.allclose(estimated_exp_shadow, pyq_exp_exact, atol=0.1)


@pytest.mark.flaky(max_runs=5)
def test_chemistry_hamiltonian_1() -> None:
    from qadence import load

    circuit = load("./tests/test_files/chem_circ.json")
    assert isinstance(circuit, QuantumCircuit)
    hamiltonian = load("./tests/test_files/chem_ham.json")
    assert isinstance(hamiltonian, AbstractBlock)
    # Restrict shadow size for faster tests.
    kwargs = {"accuracy": 0.1, "confidence": 0.1, "shadow_size": 1000}
    param_values = {"theta_0": torch.tensor([1.0])}

    model = QuantumModel(
        circuit=circuit,
        observable=hamiltonian,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    exact = model.expectation(values=param_values)
    estim = model.expectation(
        values=param_values,
        measurement=Measurements(protocol=Measurements.SHADOW, options=kwargs),
    )
    assert torch.allclose(estim, exact, atol=0.3)


@pytest.mark.flaky(max_runs=5)
def test_chemistry_hamiltonian_2() -> None:
    from qadence import load

    circuit = load("./tests/test_files/chem_circ.json")
    assert isinstance(circuit, QuantumCircuit)
    hamiltonian = ising_hamiltonian(2)
    assert isinstance(hamiltonian, AbstractBlock)
    # Restrict shadow size for faster tests.
    kwargs = {"accuracy": 0.1, "confidence": 0.1, "shadow_size": 1000}
    param_values = {"theta_0": torch.tensor([1.0])}

    model = QuantumModel(
        circuit=circuit,
        observable=hamiltonian,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    exact = model.expectation(values=param_values)
    estim = model.expectation(
        values=param_values,
        measurement=Measurements(protocol=Measurements.SHADOW, options=kwargs),
    )
    assert torch.allclose(estim, exact, atol=0.2)


def open_chem_obs() -> AbstractBlock:
    directory = os.getcwd()
    with open(os.path.join(directory, "tests/test_files/h4.json"), "r") as js:
        obs = json.loads(js.read())
    return deserialize(obs)  # type: ignore[return-value]


@pytest.mark.flaky(max_runs=5)
def test_chemistry_hamiltonian_3() -> None:
    circuit = QuantumCircuit(4, kron(Z(0), H(1), Z(2), X(3)))
    hamiltonian = open_chem_obs()
    param_values: dict = dict()

    kwargs = {"accuracy": 0.1, "confidence": 0.1, "shadow_size": 5000}

    model = QuantumModel(
        circuit=circuit,
        observable=hamiltonian,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    exact = model.expectation(values=param_values)
    estim = model.expectation(
        values=param_values,
        measurement=Measurements(protocol=Measurements.SHADOW, options=kwargs),
    )
    assert torch.allclose(estim, exact, atol=0.3)
