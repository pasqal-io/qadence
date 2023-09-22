from __future__ import annotations

from collections import Counter

import numpy as np
import numpy.typing as npt
import pytest
import torch
from braket.circuits import Circuit
from torch import Tensor

from qadence.backends import backend_factory
from qadence.backends.braket import Backend
from qadence.blocks import AbstractBlock, PrimitiveBlock
from qadence.circuit import QuantumCircuit
from qadence.constructors import ising_hamiltonian, single_z, total_magnetization
from qadence.operations import CNOT, CPHASE, RX, RY, RZ, SWAP, H, I, S, T, U, X, Y, Z, chain


def custom_obs() -> AbstractBlock:
    return X(0) * 2.0 + X(1) * 3.0 + Z(0) + Z(1) + Y(2) * 1.5 + Y(3) * 2.5


def test_register_circuit(parametric_circuit: QuantumCircuit) -> None:
    backend = Backend()
    conv_circ = backend.circuit(parametric_circuit)
    assert isinstance(conv_circ.native, Circuit)


@pytest.mark.parametrize(
    "observable",
    [
        total_magnetization(4),
        single_z(0),
        single_z(1) * 3.0,
        ising_hamiltonian(4, x_terms=np.array([0.1, 0.2, 0.3, 0.4])),
        custom_obs(),
    ],
)
def test_expectation_value(parametric_circuit: QuantumCircuit, observable: AbstractBlock) -> None:
    batch_size = 1
    values = {"x": 0.5}

    bkd = backend_factory(backend="braket", diff_mode=None)
    bra_circ, bra_obs, embed, params = bkd.convert(parametric_circuit, observable)
    expval = bkd.expectation(bra_circ, bra_obs, embed(params, values))
    assert len(expval) == batch_size


def test_expectation_value_list_of_obs(parametric_circuit: QuantumCircuit) -> None:
    batch_size = 1
    values = {"x": 0.5}  # torch.rand(batch_size)}
    observables = [ising_hamiltonian(4), total_magnetization(4), single_z(0)]
    n_obs = len(observables)

    bkd = backend_factory(backend="braket", diff_mode=None)
    bra_circ, bra_obs, embed, params = bkd.convert(parametric_circuit, observables)
    expval = bkd.expectation(bra_circ, bra_obs, embed(params, values))

    assert isinstance(expval, torch.Tensor)
    assert np.prod(expval.shape) == batch_size * n_obs
    assert torch.unique(expval).size(0) == expval.size(0)


@pytest.mark.parametrize(
    "observable, result",
    [
        ([total_magnetization(4) for _ in range(4)], torch.tensor([4.0 for _ in range(4)])),
        ([Z(k) for k in range(4)], torch.tensor([1.0 for _ in range(4)])),
    ],
)
def test_list_observables(observable: AbstractBlock, result: Tensor) -> None:
    circuit = QuantumCircuit(4, chain(Z(k) for k in range(4)))
    values = {"x": 0.5}

    bkd = backend_factory(backend="braket", diff_mode=None)
    bra_circ, bra_obs, embed, params = bkd.convert(circuit, observable)
    expval = bkd.expectation(bra_circ, bra_obs, embed(params, values))
    assert torch.allclose(expval, result)


@pytest.mark.parametrize(
    "gate, state",
    [
        (X(0), np.array([[0.0 + 0.0j, 1.0 + 0.0j]])),
        (Y(0), np.array([[0.0 + 0.0j, 0.0 + 1.0j]])),
        (Z(0), np.array([[1.0 + 0.0j, 0.0 + 0.0j]])),
        (T(0), np.array([[1.0 + 0.0j, 0.0 + 0.0j]])),
        (S(0), np.array([[1.0 + 0.0j, 0.0 + 0.0j]])),
        (H(0), 1.0 / np.sqrt(2) * np.array([1.0, 1.0])),
        (I(0), np.array([[1.0 + 0.0j, 0.0 + 0.0j]])),
    ],
)
def test_run_with_nonparametric_single_qubit_gates(
    gate: PrimitiveBlock, state: npt.NDArray
) -> None:
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    wf = backend.run(backend.circuit(circuit))
    assert np.allclose(wf, state)


@pytest.mark.parametrize(
    "parametric_gate, state",
    [
        (
            RX(0, 0.5),
            np.array(
                [[0.9689124217106447 + 0.0j, 0.0 - 0.24740395925452294j]], dtype=np.complex128
            ),
        ),
        (
            RY(0, 0.5),
            np.array(
                [[0.9689124217106447 + 0.0j, 0.24740395925452294 + 0.0j]], dtype=np.complex128
            ),
        ),
        (
            RZ(0, 0.5),
            np.array(
                [[0.9689124217106447 - 0.24740395925452294j, 0.0 + 0.0j]], dtype=np.complex128
            ),
        ),
        (
            U(0, 0.25, 0.5, 0.75),
            np.array(
                [[0.850300645292233 - 0.464521359638928j, 0.239712769302101 + 0.061208719054814j]],
                dtype=np.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_single_qubit_gates(
    parametric_gate: PrimitiveBlock, state: npt.NDArray
) -> None:
    circuit = QuantumCircuit(1, parametric_gate)
    backend = Backend()
    wf = backend.run(backend.circuit(circuit))
    assert np.allclose(wf, state)


@pytest.mark.parametrize(
    "parametric_gate, state",
    [
        (
            CNOT(0, 1),
            np.array(
                [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
                dtype=np.complex128,
            ),
        ),
        (
            X(0) * CNOT(0, 1),
            np.array(
                [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex128,
            ),
        ),
        (
            H(0) * SWAP(0, 1),
            np.array(
                [[0.70710678 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
                dtype=np.complex128,
            ),
        ),
    ],
)
def test_run_with_nonparametric_two_qubit_gates(
    parametric_gate: PrimitiveBlock, state: npt.NDArray
) -> None:
    circuit = QuantumCircuit(2, parametric_gate)
    backend = Backend()
    wf = backend.run(backend.circuit(circuit))
    assert np.allclose(wf, state)


@pytest.mark.parametrize(
    "parametric_gate, state",
    [
        (
            (X(0) @ X(1)) * CPHASE(0, 1, 0.5),
            np.array(
                [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.87758256 + 0.47942554j]],
                dtype=np.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_two_qubit_gates(
    parametric_gate: PrimitiveBlock, state: npt.NDArray
) -> None:
    circuit = QuantumCircuit(2, parametric_gate)
    backend = Backend()
    wf = backend.run(backend.circuit(circuit))
    assert np.allclose(wf, state)


@pytest.mark.parametrize(
    "gate, state",
    [
        (
            H(0),
            np.array([0.0], dtype=np.float64),
        ),
        (
            X(0),
            np.array([-1.0], dtype=np.float64),
        ),
        (
            Y(0),
            np.array([-1.0], dtype=np.float64),
        ),
        (
            Z(0),
            np.array([1.0], dtype=np.float64),
        ),
    ],
)
def test_expectation_with_pauli_gates(gate: PrimitiveBlock, state: npt.NDArray) -> None:
    circuit = QuantumCircuit(1, gate)
    observable = Z(0)
    backend = Backend()
    bra_circ, bra_obs, _, _ = backend.convert(circuit, observable)
    expectation_value = backend.expectation(bra_circ, bra_obs)
    assert np.isclose(expectation_value, state)


@pytest.mark.flaky(max_runs=5)
def test_sample_with_hadamard_gate() -> None:
    gate = H(0)
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    sample = backend.sample(backend.circuit(circuit), n_shots=10)[0]
    assert 4 <= sample["0"] <= 6
    assert 4 <= sample["1"] <= 6


@pytest.mark.parametrize(
    "gate, state",
    [
        (X(0), np.array([[["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"]]])),
        (Y(0), np.array([[["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"], ["1"]]])),
        (Z(0), np.array([[["0"], ["0"], ["0"], ["0"], ["0"], ["0"], ["0"], ["0"], ["0"], ["0"]]])),
    ],
)
def test_sample_with_pauli_gates(gate: PrimitiveBlock, state: npt.NDArray) -> None:
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    sample = backend.sample(backend.circuit(circuit), n_shots=10)[0]
    assert sample == Counter(state.flatten())
