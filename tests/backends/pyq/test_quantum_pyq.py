from __future__ import annotations

import random
from collections import Counter
from itertools import product
from typing import Callable

import numpy as np
import pytest
import strategies as st
import torch
from hypothesis import given, settings
from pyqtorch import U as pyqU
from pyqtorch import zero_state as pyq_zero_state
from pyqtorch.circuit import QuantumCircuit as PyQQuantumCircuit
from sympy import acos
from torch import Tensor

from qadence.backends import backend_factory
from qadence.backends.pyqtorch.backend import Backend
from qadence.backends.pyqtorch.config import Configuration as PyqConfig
from qadence.blocks import (
    AbstractBlock,
    PrimitiveBlock,
    chain,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors import (
    hea,
    ising_hamiltonian,
    total_magnetization,
    zz_hamiltonian,
)
from qadence.models import QuantumModel
from qadence.operations import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    RX,
    RY,
    RZ,
    SWAP,
    AnalogSWAP,
    H,
    HamEvo,
    I,
    S,
    T,
    U,
    X,
    Y,
    Z,
)
from qadence.parameters import FeatureParameter, Parameter
from qadence.states import random_state, uniform_state, zero_state
from qadence.transpile import set_trainable
from qadence.types import PI, BackendName, DiffMode


def custom_obs() -> AbstractBlock:
    return X(0) * 2.0 + X(1) * 3.0 + Z(0) + Z(1) + Y(2) * 1.5 + Y(3) * 2.5


def test_register_circuit(parametric_circuit: QuantumCircuit) -> None:
    backend = Backend()
    conv_circ = backend.circuit(parametric_circuit)
    assert len(conv_circ.native.operations) > 0


def wf_is_normalized(wf: torch.Tensor) -> torch.Tensor:
    return torch.isclose(sum(torch.flatten(torch.abs(wf) ** 2)), torch.tensor(1.00))


@pytest.mark.parametrize(
    "observable",
    [
        total_magnetization(4),
        # single_z(0), # FIXME: enable those again
        # single_z(1) * 3.0,
        ising_hamiltonian(4, x_terms=np.array([0.1, 0.2, 0.3, 0.4])),
        custom_obs(),
    ],
)
def test_expectation_value(parametric_circuit: QuantumCircuit, observable: AbstractBlock) -> None:
    # TODO: refactor parametric_circuit fixture
    circuit = QuantumCircuit(parametric_circuit.n_qubits, parametric_circuit.block)

    batch_size = 10
    values = {"x": torch.rand(batch_size)}

    bkd = Backend()
    pyqtorch_circ, pyqtorch_obs, embed, params = bkd.convert(parametric_circuit, observable)
    expval = bkd.expectation(pyqtorch_circ, pyqtorch_obs, embed(params, values))
    assert len(expval) == batch_size


@pytest.mark.parametrize(
    "observable, result",
    [
        ([total_magnetization(4) for _ in range(4)], torch.tensor([4.0 for _ in range(4)])),
        ([Z(k) for k in range(4)], torch.tensor([1.0 for _ in range(4)])),
    ],
)
def test_list_observables(observable: AbstractBlock, result: Tensor) -> None:
    circuit = QuantumCircuit(4, chain(Z(k) for k in range(4)))

    bkd = backend_factory(backend="pyqtorch", diff_mode=None)
    bra_circ, bra_obs, embed, params = bkd.convert(circuit, observable)
    expval = bkd.expectation(bra_circ, bra_obs, embed(params, {}))
    assert torch.allclose(expval, result)


@pytest.mark.parametrize("n_obs, loop_expectation", product([1, 2, 3], [True, False]))
def test_list_observables_with_batches(n_obs: int, loop_expectation: bool) -> None:
    n_qubits = 4
    x = FeatureParameter("x")
    circuit = QuantumCircuit(n_qubits, kron(RX(k, x) for k in range(n_qubits)))

    observables = []
    for i in range(3):
        o = float(i + 1) * ising_hamiltonian(4)
        observables.append(o)

    observables = observables[:n_obs]
    batch_size = 10
    values = {"x": torch.rand(batch_size)}

    model = QuantumModel(circuit, observables, configuration={"loop_expectation": loop_expectation})
    expval = model.expectation(values)

    assert len(expval.shape) == 2 and expval.shape[0] == batch_size and expval.shape[1] == n_obs
    factors = torch.linspace(1, n_obs, n_obs)
    for i, e in enumerate(expval):
        tmp = torch.div(e, factors * e[0])
        assert torch.allclose(tmp, torch.ones(n_obs))


@pytest.mark.parametrize("n_shots", [5, 10, 100, 1000, 10000])
def test_sample(parametric_circuit: QuantumCircuit, n_shots: int) -> None:
    batch_size = 10
    values = {"x": torch.rand(batch_size)}

    bkd = Backend()
    pyqtorch_circ, _, embed, params = bkd.convert(parametric_circuit)
    samples = bkd.sample(pyqtorch_circ, embed(params, values), n_shots=n_shots)
    assert len(samples) == batch_size  # type: ignore[arg-type]
    assert all(sum(s.values()) == n_shots for s in samples)


@pytest.mark.xfail(reason="Removed params from native_circuit")
def test_native_circuit(parametric_circuit: QuantumCircuit) -> None:
    backend = Backend()
    conv_circ = backend.circuit(parametric_circuit)
    assert isinstance(conv_circ.native, PyQQuantumCircuit)
    assert len([p for p in conv_circ.native.parameters()]) == parametric_circuit.num_parameters


def test_raise_error_for_ill_dimensioned_initial_state() -> None:
    circuit = QuantumCircuit(2, X(0) @ X(1))
    backend = Backend()
    initial_state = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    with pytest.raises(ValueError):
        backend.run(backend.circuit(circuit), state=initial_state)


@pytest.mark.parametrize(
    "gate, state",
    [
        (X(0), torch.tensor([0.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex128)),
        (Y(0), torch.tensor([0.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex128)),
        (Z(0), torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)),
        (T(0), torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)),
        (S(0), torch.tensor([1.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)),
        (H(0), torch.tensor(1.0 / np.sqrt(2) * np.array([1.0, 1.0]), dtype=torch.complex128)),
    ],
)
def test_run_with_nonparametric_single_qubit_gates(
    gate: PrimitiveBlock, state: torch.Tensor
) -> None:
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    pyqtorch_circ = backend.circuit(circuit)
    wf = backend.run(pyqtorch_circ)
    assert torch.allclose(wf, state)
    # Same test by passing explicitly the initial state.
    initial_state = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
    wf = backend.run(pyqtorch_circ, state=initial_state)
    assert torch.allclose(wf, state)


@pytest.mark.parametrize(
    "gate, matrix",
    [
        (
            X(0),
            torch.tensor(
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            Y(0),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            Z(0),
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            T(0),
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, np.exp((PI / 4.0) * 1j)]],
                dtype=torch.complex128,
            ),
        ),
        (
            S(0),
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.complex128
            ),
        ),
        (
            H(0),
            torch.tensor(
                1.0 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]]), dtype=torch.complex128
            ),
        ),
    ],
)
def test_run_with_nonparametric_single_qubit_gates_and_random_initial_state(
    gate: PrimitiveBlock, matrix: torch.Tensor
) -> None:
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    theta1 = random.uniform(0.0, 2.0 * PI)
    complex1 = complex(np.cos(theta1), np.sin(theta1))
    theta2 = random.uniform(0.0, 2.0 * PI)
    complex2 = complex(np.cos(theta2), np.sin(theta2))
    initial_state = torch.tensor([[complex1, complex2]], dtype=torch.complex128)
    wf = backend.run(backend.circuit(circuit), state=initial_state)
    expected_state = torch.matmul(matrix, initial_state[0])
    assert torch.allclose(wf, expected_state)


@pytest.mark.parametrize(
    "parametric_gate, state",
    [
        (
            RX(0, 0.5),
            torch.tensor(
                [[0.9689124217106447 + 0.0j, 0.0 - 0.24740395925452294j]], dtype=torch.complex128
            ),
        ),
        (
            RY(0, 0.5),
            torch.tensor(
                [[0.9689124217106447 + 0.0j, 0.24740395925452294 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            RZ(0, 0.5),
            torch.tensor(
                [[0.9689124217106447 - 0.24740395925452294j, 0.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            U(0, 0.25, 0.5, 0.75),
            torch.tensor(
                [[0.850300645292233 - 0.464521359638928j, 0.239712769302101 + 0.061208719054814j]],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_single_qubit_gates(
    parametric_gate: PrimitiveBlock, state: torch.Tensor
) -> None:
    circuit = QuantumCircuit(1, parametric_gate)
    backend = Backend()
    pyqtorch_circ, _, embed, params = backend.convert(circuit)
    wf = backend.run(pyqtorch_circ, embed(params, {}))
    assert torch.allclose(wf, state)


def test_ugate_pure_pyqtorch() -> None:
    thetas = torch.rand(3)
    pyq_state = pyq_zero_state(n_qubits=1)
    Qadence_u = U(0, phi=thetas[0], theta=thetas[1], omega=thetas[2])
    circ = QuantumCircuit(1, Qadence_u)
    backend = Backend()
    convert = backend.convert(circ)
    values = convert.embedding_fn(convert.params, {})
    Qadence_state = backend.run(convert.circuit, values)
    pyqtorch_u = pyqU(0, *values.keys())
    f_state = torch.reshape(pyqtorch_u(pyq_state, values), (1, 2))
    assert torch.allclose(f_state, Qadence_state)


theta = 0.5
theta_half = theta / 2.0


@pytest.mark.parametrize(
    "parametric_gate, matrix",
    [
        (
            RX(0, theta),
            torch.tensor(
                [
                    [np.cos(theta_half), -np.sin(theta_half) * 1j],
                    [-np.sin(theta_half) * 1j, np.cos(theta_half)],
                ],
                dtype=torch.complex128,
            ),
        ),
        (
            RY(0, theta),
            torch.tensor(
                [
                    [np.cos(theta_half), -np.sin(theta_half)],
                    [np.sin(theta_half), np.cos(theta_half)],
                ],
                dtype=torch.complex128,
            ),
        ),
        (
            RZ(0, 0.5),
            torch.tensor(
                [[np.exp(-1j * theta_half), 0.0], [0.0, np.exp(1j * theta_half)]],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_single_qubit_gates_and_random_initial_state(
    parametric_gate: PrimitiveBlock, matrix: torch.Tensor
) -> None:
    circuit = QuantumCircuit(1, parametric_gate)
    backend = Backend()
    pyqtorch_circ, _, embed, params = backend.convert(circuit)
    theta1 = random.uniform(0.0, 2.0 * PI)
    complex1 = complex(np.cos(theta1), np.sin(theta1))
    theta2 = random.uniform(0.0, 2.0 * PI)
    complex2 = complex(np.cos(theta2), np.sin(theta2))
    initial_state = torch.tensor([[complex1, complex2]], dtype=torch.complex128)
    wf = backend.run(pyqtorch_circ, embed(params, {}), state=initial_state)
    expected_state = torch.matmul(matrix, initial_state[0])
    assert torch.allclose(wf, expected_state)


@pytest.mark.parametrize(
    "parametric_gate, state",
    [
        (
            CRX(0, 1, 0.5),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.24740395925452294j, 0.9689124217106447 + 0.0j]],
                dtype=torch.complex128,
            ),
        ),
        (
            CRY(0, 1, 0.5),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 + 0.0j, -0.24740395925452294 + 0.0j, 0.9689124217106447 + 0.0j]],
                dtype=torch.complex128,
            ),
        ),
        (
            CRZ(0, 1, 0.5),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.9689124217106447 + 0.24740395925452294j]],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_two_qubit_gates(
    parametric_gate: PrimitiveBlock, state: torch.Tensor
) -> None:
    circuit = QuantumCircuit(2, parametric_gate)
    backend = Backend()
    pyqtorch_circ, _, embed, params = backend.convert(circuit)
    # Initialising the state to |11> to produce non-trivial outputs.
    initial_state = torch.tensor(
        [[0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.complex128
    )
    wf = backend.run(pyqtorch_circ, embed(params, {}), state=initial_state)
    assert torch.allclose(wf, state)


@pytest.mark.parametrize(
    "parametric_gate, matrix",
    [
        (
            CRX(0, 1, theta),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, np.cos(theta_half), -np.sin(theta_half) * 1j],
                    [0.0, 0.0, -np.sin(theta_half) * 1j, np.cos(theta_half)],
                ],
                dtype=torch.complex128,
            ),
        ),
        (
            CRY(0, 1, 0.5),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, np.cos(theta_half), -np.sin(theta_half)],
                    [0.0, 0.0, np.sin(theta_half), np.cos(theta_half)],
                ],
                dtype=torch.complex128,
            ),
        ),
        (
            CRZ(0, 1, 0.5),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, np.exp(-1j * theta_half), 0.0],
                    [0.0, 0.0, 0.0, np.exp(1j * theta_half)],
                ],
                dtype=torch.complex128,
            ),
        ),
    ],
)
def test_run_with_parametric_two_qubit_gates_and_random_state(
    parametric_gate: PrimitiveBlock, matrix: torch.Tensor
) -> None:
    circuit = QuantumCircuit(2, parametric_gate)
    backend = Backend()

    pyqtorch_circ, _, embed, params = backend.convert(circuit)

    # Initialising random state to produce non-trivial outputs.
    random_coefs = np.array([random.uniform(0, 1) for _ in range(8)])
    random_coefs = random_coefs / np.sqrt(np.sum(np.square(random_coefs)))
    initial_state = torch.tensor(
        [
            [
                random_coefs[0] + random_coefs[1] * 1j,
                random_coefs[2] + random_coefs[3] * 1j,
                random_coefs[4] + random_coefs[5] * 1j,
                random_coefs[6] + random_coefs[7] * 1j,
            ],
        ],
        dtype=torch.complex128,
    )
    wf = backend.run(pyqtorch_circ, embed(params, {}), state=initial_state)
    expected_state = torch.matmul(matrix, initial_state[0])
    assert torch.allclose(wf, expected_state)


@pytest.mark.parametrize(
    "gate, state",
    [
        (
            H(0),
            torch.tensor([[0.0]]),
        ),
        (
            X(0),
            torch.tensor([[-1.0]]),
        ),
        (
            Y(0),
            torch.tensor([[-1.0]]),
        ),
        (
            Z(0),
            torch.tensor([[1.0]]),
        ),
    ],
)
def test_expectation_with_pauli_gates(gate: PrimitiveBlock, state: torch.Tensor) -> None:
    circuit = QuantumCircuit(1, gate)
    observable = Z(0)
    backend = Backend()
    pyqtorch_circ, pyqtorch_obs, embed, params = backend.convert(circuit, observable)
    expectation_value = backend.expectation(pyqtorch_circ, pyqtorch_obs, embed(params, {}))
    assert expectation_value == state


@pytest.mark.parametrize(
    "gate, matrix",
    [
        (
            H(0),
            torch.tensor(
                1.0 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]]), dtype=torch.complex128
            ),
        ),
        (
            X(0),
            torch.tensor(
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            Y(0),
            torch.tensor(
                [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
        (
            Z(0),
            torch.tensor(
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.complex128
            ),
        ),
    ],
)
def test_expectation_with_pauli_gates_and_random_state(
    gate: PrimitiveBlock, matrix: torch.Tensor
) -> None:
    circuit = QuantumCircuit(1, gate)
    observable = Z(0)
    backend = Backend()
    pyqtorch_circ, pyqtorch_obs, embed, params = backend.convert(circuit, observable)

    theta1 = random.uniform(0.0, 2.0 * PI)
    complex1 = complex(np.cos(theta1), np.sin(theta1))
    theta2 = random.uniform(0.0, 2.0 * PI)
    complex2 = complex(np.cos(theta2), np.sin(theta2))
    initial_state = torch.tensor([[complex1, complex2]], dtype=torch.complex128)
    expectation_value = backend.expectation(
        pyqtorch_circ, pyqtorch_obs, embed(params, {}), state=initial_state
    )
    Z_matrix = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.complex128
    )
    final_state = torch.matmul(Z_matrix, torch.matmul(matrix, initial_state[0]))
    probas = torch.square(torch.abs(final_state))
    expected_value = probas[0] - probas[1]
    assert torch.allclose(expectation_value, expected_value)


@pytest.mark.flaky(max_runs=5)
def test_sample_with_hadamard_gate() -> None:
    gate = H(0)
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    conv = backend.convert(circuit)
    samples = backend.sample(conv.circuit, n_shots=100)
    assert len(samples) == 1
    sample = samples[0]
    assert 40 <= sample["0"] <= 60
    assert 40 <= sample["1"] <= 60


@pytest.mark.parametrize(
    "gate, counter",
    [
        (X(0), [Counter({"1": 10})]),
        (Y(0), [Counter({"1": 10})]),
        (Z(0), [Counter({"0": 10})]),
    ],
)
def test_sample_with_pauli_gates(gate: PrimitiveBlock, counter: Counter) -> None:
    circuit = QuantumCircuit(1, gate)
    backend = Backend()
    sample = backend.sample(backend.circuit(circuit), n_shots=10)
    assert sample == counter


def test_controlled_rotation_gates_with_heterogeneous_parameters() -> None:
    from qadence.parameters import FeatureParameter

    block = CRX(0, 1, 0.5) * CRY(1, 2, FeatureParameter("x")) * CRZ(2, 3, "y")
    circ = QuantumCircuit(4, block)
    backend = Backend()
    conv = backend.convert(circ)

    values = {"x": torch.rand(2)}
    wf = backend.run(conv.circuit, conv.embedding_fn(conv.params, values))
    assert wf.size() == (2, 2**4)


@pytest.mark.parametrize(
    "block",
    [
        X(0),
        RZ(1, 0.5),
        # CRY(0,1,0.2) write proper test for this
    ],
)
def test_scaled_operation(block: AbstractBlock) -> None:
    backend = Backend()
    state = torch.rand(1, 4, dtype=torch.cdouble)

    circ = QuantumCircuit(2, block)
    pyqtorch_circ, _, embed, params = backend.convert(circ)
    wf = backend.run(pyqtorch_circ, embed(params, {}), state=state)

    circ = QuantumCircuit(2, block * 2)
    pyqtorch_circ, _, embed, params = backend.convert(circ)
    wf2 = backend.run(pyqtorch_circ, embed(params, {}), state=state)

    assert torch.allclose(wf * 2, wf2)


@pytest.mark.parametrize("batch_size", [i for i in range(10)])
def test_scaled_featureparam_batching(batch_size: int) -> None:
    backend = Backend()
    block = FeatureParameter("x") * X(0)
    circ = QuantumCircuit(1, block)
    pyqtorch_circ, _, embed, params = backend.convert(circ)
    rand_vals = torch.rand(batch_size)
    param_values = embed(params, {"x": rand_vals})
    wf = backend.run(pyqtorch_circ, param_values)
    wf2 = backend.run(pyqtorch_circ, embed(params, {"x": torch.ones(batch_size)}))
    assert torch.allclose(wf, wf2 * rand_vals.unsqueeze(1))


@pytest.mark.parametrize(
    "block",
    [
        X(0),
        Y(0),
        Z(0),
        # S(0), # TODO implement SDagger in PyQ
        # T(0), # TODO implement TDagger in PyQ
        CNOT(0, 1),
        # CZ(0, 1), # TODO implement CZ in PyQ?
        SWAP(0, 1),
        H(0),
        I(0),
        # Zero(), # TODO what to test here?
        chain(X(0), Y(0), Z(0), Y(0)),
        kron(X(1), Y(3), Z(4), Y(2)),
        chain(kron(X(0), Y(1)), kron(Z(3), H(1))),
        chain(CNOT(0, 1), CNOT(1, 0)),
    ],
)
def test_dagger_returning_fixed_gates(block: AbstractBlock) -> None:
    nqubits = block.n_qubits
    circ = QuantumCircuit(nqubits, block, block.dagger())
    backend = Backend()
    conv = backend.convert(circ)
    initial_state = torch.rand((1, 2**nqubits), dtype=torch.cdouble) + 1j * torch.rand(
        (1, 2**nqubits), dtype=torch.cdouble
    )
    initial_state = initial_state / torch.sqrt(sum(abs(initial_state) ** 2))
    wf = backend.run(conv.circuit, state=initial_state)
    assert torch.allclose(wf, initial_state)


@pytest.mark.parametrize(
    "block_class",
    [
        RX,
        RY,
        RZ,
        CRX,
        CRY,
        CRZ,
        CPHASE,
        AnalogSWAP,
    ],
)
@pytest.mark.parametrize("p_type", [0.52, "x", Parameter("x"), acos(Parameter("x"))])
@pytest.mark.parametrize("trainable", [True, False])
def test_dagger_returning_parametric_gates(
    block_class: AbstractBlock, p_type: float | str | Parameter, trainable: bool
) -> None:
    n_qubits = 2 if block_class not in [RX, RY, RZ] else 1
    block = block_class(*tuple(range(n_qubits)), p_type)  # type: ignore[operator]
    set_trainable(block, trainable)
    circ = QuantumCircuit(n_qubits, block, block.dagger())
    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)
    run_params = embed(params, {"x": torch.tensor([0.52])})
    initial_state = torch.rand((1, 2**n_qubits), dtype=torch.cdouble) + 1j * torch.rand(
        (1, 2**n_qubits), dtype=torch.cdouble
    )
    initial_state = initial_state / torch.sqrt(sum(abs(initial_state) ** 2))
    wf = backend.run(pyqtorch_circ, run_params, state=initial_state.clone())
    assert torch.allclose(wf, initial_state)


def test_dagger_returning_kernel() -> None:
    generatorx = 3.1 * X(0) + 1.2 * Y(0) + 1.1 * Y(1) + 1.9 * X(1) + 2.4 * Z(0) * Z(1)
    fmx = HamEvo(generatorx, parameter=acos(Parameter("x")))
    set_trainable(fmx, False)
    fmy = HamEvo(generatorx, parameter=acos(Parameter("y")))
    set_trainable(fmy, False)
    ansatz = hea(2, 2)
    set_trainable(ansatz, True)
    circ = QuantumCircuit(2, fmx, ansatz.dagger(), ansatz, fmy.dagger())
    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)

    initial_state = torch.rand((1, 2**2), dtype=torch.cdouble) + 1j * torch.rand(
        (1, 2**2), dtype=torch.cdouble
    )
    initial_state = initial_state / torch.sqrt(4 * sum(abs(initial_state) ** 2))

    run_params = embed(params, {"x": torch.tensor([0.52]), "y": torch.tensor(0.52)})
    wf = backend.run(pyqtorch_circ, run_params, state=initial_state.clone())
    assert wf_is_normalized(wf)
    assert torch.allclose(wf, initial_state)

    run_params = embed(params, {"x": torch.tensor([0.38]), "y": torch.tensor(0.92)})
    wf = backend.run(pyqtorch_circ, run_params, state=initial_state.clone())
    assert not torch.allclose(wf, initial_state)


def test_scaled_blocks() -> None:
    circuit = QuantumCircuit(1, 3.1 * (X(0) + X(0)))
    model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    wf = model.run({})
    assert isinstance(wf, torch.Tensor)

    circuit = QuantumCircuit(2, 2 * (X(0) @ X(1)))
    model = QuantumModel(circuit, diff_mode=DiffMode.AD)
    wf = model.run({})
    assert isinstance(wf, torch.Tensor)


def test_kron_chain_add_circuit() -> None:
    p0 = I(0) * 0.5 + Z(0) * 0.5
    p1 = I(0) * 0.5 + Z(0) * (-0.5)
    cnot = kron(p0, I(1)) + kron(p1, X(1))

    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=None)

    circ = QuantumCircuit(2, chain(X(0), X(1), cnot))
    (circ_conv, _, embedding_fn, params) = backend.convert(circ)
    res_constructed = backend.run(circ_conv, embedding_fn(params, {}))

    circ = QuantumCircuit(2, chain(X(0), X(1), CNOT(0, 1)))
    (circ_conv, _, embedding_fn, params) = backend.convert(circ)
    res_native = backend.run(circ_conv, embedding_fn(params, {}))

    assert torch.allclose(res_constructed, res_native)


def test_swap_equivalences() -> None:
    block = AnalogSWAP(0, 1)
    ref_block = SWAP(0, 1)
    circ = QuantumCircuit(2, block, ref_block)
    state_r = torch.rand(2**2, dtype=torch.cdouble) - 0.5
    state_i = torch.rand(2**2, dtype=torch.cdouble) - 0.5
    norm = torch.linalg.vector_norm(state_r + 1j * state_i)
    wf_init = torch.Tensor(((state_r + 1j * state_i) / norm).unsqueeze(0))
    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=None)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)
    run_params = embed(params, {})
    wf = backend.run(pyqtorch_circ, run_params, state=wf_init.clone())

    # check equivalence up to rotation
    angle = torch.angle(wf_init[0, 0]).detach()
    wf_init = wf_init * torch.exp(-1j * angle)
    angle = torch.angle(wf[0, 0]).detach()
    wf = wf * torch.exp(-1j * angle)
    assert torch.allclose(wf, wf_init)


@given(st.batched_digital_circuits())
@settings(deadline=None)
def test_batched_circuits(
    circuit_and_inputs: tuple[QuantumCircuit, dict[str, torch.Tensor]]
) -> None:
    circuit, inputs = circuit_and_inputs
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    (circ_pyqtorch, _, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs))
    assert not torch.any(torch.isnan(wf_pyqtorch))


@pytest.mark.parametrize("diff_mode", [DiffMode.GPSR, DiffMode.AD, DiffMode.ADJOINT])
@pytest.mark.parametrize("obs", [total_magnetization, zz_hamiltonian])
@given(st.batched_digital_circuits())
@settings(deadline=None)
def test_sparse_obs_expectation_value(
    diff_mode: DiffMode,
    obs: Callable,
    circuit_and_inputs: tuple[QuantumCircuit, dict[str, torch.Tensor]],
) -> None:
    non_sparse_cfg = PyqConfig()
    non_sparse_cfg.use_sparse_observable = False
    sparse_cfg = PyqConfig()
    sparse_cfg.use_sparse_observable = True
    circuit, inputs = circuit_and_inputs
    observable = obs(circuit.n_qubits)
    qm_nonsparse = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
        configuration=non_sparse_cfg,
    )
    qm_sparse = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
        configuration=sparse_cfg,
    )
    expval = qm_nonsparse.expectation(inputs)
    expval_s = qm_sparse.expectation(inputs)

    assert torch.allclose(expval, expval_s)


@pytest.mark.parametrize("state_fn", [uniform_state, random_state, zero_state])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("obs", [total_magnetization, zz_hamiltonian])
@given(st.batched_digital_circuits())
@settings(deadline=None)
def test_move_to_dtype(
    state_fn: Callable,
    dtype: torch.dtype,
    obs: Callable,
    circuit_and_inputs: tuple[QuantumCircuit, dict[str, torch.Tensor]],
) -> None:
    circuit, inputs = circuit_and_inputs
    observable = obs(circuit.n_qubits)
    qm = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
    )
    qm = qm.to(dtype=dtype)
    inputs = {k: v.to(dtype=dtype) for k, v in inputs.items()}
    assert qm._circuit.native.dtype == dtype
    assert all([obs.native.dtype == dtype for obs in qm._observable])  # type: ignore[union-attr]
    state = state_fn(circuit.n_qubits)
    state = state.to(dtype=dtype)
    assert state.dtype == dtype
    # breakpoint()
    wf = qm.run(inputs, state=state)
    assert wf.dtype == dtype
    expval = qm.expectation(inputs, state=state)
    assert expval.dtype == torch.float64 if dtype == torch.cdouble else torch.float32
