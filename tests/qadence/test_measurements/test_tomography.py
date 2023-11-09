from __future__ import annotations

from collections import Counter
from typing import List

import pytest
import strategies as st  # type: ignore
from hypothesis import given, settings
from metrics import HIGH_ACCEPTANCE, LOW_ACCEPTANCE, MIDDLE_ACCEPTANCE  # type: ignore
from torch import allclose, autograd, flatten, manual_seed, ones_like, rand, tensor

from qadence.backends import backend_factory
from qadence.blocks import (
    AbstractBlock,
    add,
    chain,
    kron,
)
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.constructors import (
    feature_map,
    hea,
    total_magnetization,
    zz_hamiltonian,
)
from qadence.measurements import Measurements
from qadence.measurements.tomography import (
    compute_expectation as basic_tomography,
)
from qadence.measurements.tomography import (
    empirical_average,
    get_counts,
    get_qubit_indices_for_op,
    iterate_pauli_decomposition,
    rotate,
)
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN, QuantumModel
from qadence.operations import RX, RY, H, SDagger, X, Y, Z
from qadence.parameters import Parameter
from qadence.types import BackendName, BasisSet, DiffMode

manual_seed(1)

BACKENDS = ["pyqtorch", "braket"]
DIFF_MODE = ["ad", "gpsr"]


@pytest.mark.parametrize(
    "pauli_word, exp_indices_X, exp_indices_Y",
    [
        (kron(X(0), X(1)), [[0, 1]], [[]]),
        (kron(X(0), Y(1)), [[0]], [[1]]),
        (kron(Y(0), Y(1)), [[]], [[0, 1]]),
        (kron(Z(0), Z(1)), [[]], [[]]),
        (add(X(0), X(1)), [[0], [1]], [[], []]),
        (add(X(0), Y(1)), [[0], []], [[], [1]]),
        (add(Y(0), Y(1)), [[], []], [[0], [1]]),
        (add(Z(0), Z(1)), [[], []], [[], []]),
        (add(kron(X(0), Z(2)), 1.5 * kron(Y(1), Z(2))), [[0], []], [[], [1]]),
        (
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
            [[0, 2], [], [1, 3]],
            [[1, 3], [0, 2], []],
        ),
    ],
)
def test_get_qubit_indices_for_op(
    pauli_word: tuple, exp_indices_X: list, exp_indices_Y: list
) -> None:
    pauli_decomposition = unroll_block_with_scaling(pauli_word)

    indices_X = []
    indices_Y = []
    for index, pauli_term in enumerate(pauli_decomposition):
        indices_x = get_qubit_indices_for_op(pauli_term, X(0))
        # if indices_x:
        indices_X.append(indices_x)
        indices_y = get_qubit_indices_for_op(pauli_term, Y(0))
        # if indices_y:
        indices_Y.append(indices_y)
    assert indices_X == exp_indices_X
    assert indices_Y == exp_indices_Y


@pytest.mark.parametrize(
    "circuit, observable, expected_circuit",
    [
        (
            QuantumCircuit(2, kron(X(0), X(1))),
            kron(X(0), Z(2)) + 1.5 * kron(Y(1), Z(2)),
            [
                QuantumCircuit(2, chain(kron(X(0), X(1)), Z(0) * H(0))),
                QuantumCircuit(2, chain(kron(X(0), X(1)), SDagger(1) * H(1))),
            ],
        ),
        (
            QuantumCircuit(4, kron(X(0), X(1), X(2), X(3))),
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
            [
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        Z(0) * H(0),
                        Z(2) * H(2),
                        SDagger(1) * H(1),
                        SDagger(3) * H(3),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        SDagger(0) * H(0),
                        SDagger(2) * H(2),
                    ),
                ),
                QuantumCircuit(
                    4,
                    chain(
                        kron(X(0), X(1), X(2), X(3)),
                        Z(1) * H(1),
                        Z(3) * H(3),
                    ),
                ),
            ],
        ),
    ],
)
def test_rotate(
    circuit: QuantumCircuit,
    observable: AbstractBlock,
    expected_circuit: List[QuantumCircuit],
) -> None:
    pauli_decomposition = unroll_block_with_scaling(observable)
    for index, pauli_term in enumerate(pauli_decomposition):
        rotated_circuit = rotate(circuit, pauli_term)
        assert rotated_circuit == expected_circuit[index]


def test_raise_errors() -> None:
    block = H(0)
    observable = Z(0)
    circuit = QuantumCircuit(1, block)
    pyqtorch_backend = backend_factory(BackendName.PYQTORCH, diff_mode=None)
    (conv_circ, _, _, _) = pyqtorch_backend.convert(circuit, observable)
    options = {"n_shots": 10000}
    with pytest.raises(TypeError):
        basic_tomography(
            circuit=conv_circ,
            param_values={},
            observables=observable,  # type: ignore[arg-type]
            options=options,
        )
    options = {"shots": 10000}
    with pytest.raises(KeyError):
        basic_tomography(
            circuit=conv_circ, param_values={}, observables=[observable], options=options
        )


def test_get_counts() -> None:
    samples = [Counter({"00": 10, "01": 50, "10": 20, "11": 20})]
    support = [0]
    counts = get_counts(samples, support)
    assert counts == [Counter({"0": 60, "1": 40})]
    support = [1]
    counts = get_counts(samples, support)
    assert counts == [Counter({"0": 30, "1": 70})]
    support = [0, 1]
    counts = get_counts(samples, support)
    assert counts == samples

    samples = [
        Counter(
            {
                "1111": 1653,
                "0000": 1586,
                "0001": 1463,
                "0110": 1286,
                "1110": 998,
                "0101": 668,
                "0111": 385,
                "1000": 327,
                "0011": 322,
                "1100": 281,
                "1001": 218,
                "1010": 213,
                "0100": 187,
                "1101": 172,
                "1011": 154,
                "0010": 87,
            }
        )
    ]
    support = [0, 1, 2, 3]
    counts = get_counts(samples, support)
    assert counts == samples


def test_empirical_average() -> None:
    samples = [Counter({"00": 10, "01": 50, "10": 20, "11": 20})]
    support = [0]
    assert allclose(empirical_average(samples, support), tensor([0.2]))
    support = [1]
    assert allclose(empirical_average(samples, support), tensor([-0.4]))
    support = [0, 1]
    assert allclose(empirical_average(samples, support), tensor([-0.4]))
    samples = [
        Counter(
            {
                "1111": 1653,
                "0000": 1586,
                "0001": 1463,
                "0110": 1286,
                "1110": 998,
                "0101": 668,
                "0111": 385,
                "1000": 327,
                "0011": 322,
                "1100": 281,
                "1001": 218,
                "1010": 213,
                "0100": 187,
                "1101": 172,
                "1011": 154,
                "0010": 87,
            }
        )
    ]
    support = [0, 1, 2, 3]
    assert allclose(empirical_average(samples, support), tensor([0.2454]))


# Disable cases are passing at the expense of high (1 billion) nshots.
# To keep reasonable run time, less expensive cases are tested.
# Some observables also contain ScaleBlock for which PSR are not defined.
@pytest.mark.parametrize(
    "circuit, values, observable",
    [
        (QuantumCircuit(1, H(0)), {}, Z(0)),
        (QuantumCircuit(2, kron(H(0), H(1))), {}, kron(X(0), X(1))),
        (
            QuantumCircuit(4, feature_map(4, fm_type=BasisSet.CHEBYSHEV), hea(4, depth=2)),
            {"phi": rand(1)},
            total_magnetization(4),
        ),
        (
            QuantumCircuit(4, feature_map(4, fm_type=BasisSet.CHEBYSHEV), hea(4, depth=2)),
            {"phi": rand(1)},
            zz_hamiltonian(4),
        ),
        # (
        #     QuantumCircuit(4, feature_map(4, fm_type=BasisSet.CHEBYSHEV), hea(4, depth=2)),
        #     {"phi": rand(1)},
        #     ising_hamiltonian(4),
        #     HIGH_ACCEPTANCE,
        # ),
        (
            QuantumCircuit(4, feature_map(4, fm_type=BasisSet.CHEBYSHEV), hea(4, depth=2)),
            {"phi": rand(1)},
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
        ),
    ],
)
def test_iterate_pauli_decomposition(
    circuit: QuantumCircuit,
    values: dict,
    observable: AbstractBlock,
) -> None:
    pauli_decomposition = unroll_block_with_scaling(observable)
    pyqtorch_backend = backend_factory(BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    (conv_circ, conv_obs, embed, params) = pyqtorch_backend.convert(circuit, observable)
    param_values = embed(params, values)
    pyqtorch_expectation = pyqtorch_backend.expectation(conv_circ, conv_obs, param_values)[0]
    estimated_values = iterate_pauli_decomposition(
        circuit=conv_circ.abstract,
        param_values=param_values,
        pauli_decomposition=pauli_decomposition,
        n_shots=1000000,
    )
    assert allclose(estimated_values, pyqtorch_expectation, atol=LOW_ACCEPTANCE)


@given(st.digital_circuits())
@settings(deadline=None)
def test_basic_tomography_direct_call(circuit: QuantumCircuit) -> None:
    observable = Z(0) ^ circuit.n_qubits
    pyqtorch_backend = backend_factory(BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    (conv_circ, conv_obs, embed, params) = pyqtorch_backend.convert(circuit, observable)
    inputs = rand_featureparameters(circuit, 1)
    kwargs = {"n_shots": 100000}
    tomo_values = basic_tomography(
        conv_circ.abstract, [c_o.abstract for c_o in conv_obs], embed(params, inputs), kwargs
    )[0]
    estimated_values = flatten(tomo_values)

    pyqtorch_expectation = pyqtorch_backend.expectation(conv_circ, conv_obs, embed(params, inputs))[
        0
    ]
    assert allclose(estimated_values, pyqtorch_expectation, atol=LOW_ACCEPTANCE)


@given(st.restricted_circuits())
@settings(deadline=None)
def test_basic_tomography_for_backend_forward_pass(circuit: QuantumCircuit) -> None:
    obs = Z(0) ^ circuit.n_qubits
    kwargs = {"n_shots": 100000}
    for backend in BACKENDS:
        for diff_mode in [DiffMode.GPSR]:
            inputs = rand_featureparameters(circuit, 1)
            qm = QuantumModel(circuit=circuit, observable=obs, backend=backend, diff_mode=diff_mode)
            exp_tomo = qm.expectation(
                values=inputs,
                measurement=Measurements(
                    protocol=Measurements.TOMOGRAPHY,
                    options=kwargs,
                ),
            )[0]
            estimated_values = flatten(exp_tomo)
            expectation_values = qm.expectation(values=inputs)[0]
            assert allclose(estimated_values, expectation_values, atol=LOW_ACCEPTANCE)


@given(st.digital_circuits())
@settings(deadline=None)
def test_basic_tomography_for_quantum_model(circuit: QuantumCircuit) -> None:
    backend = BackendName.PYQTORCH
    diff_mode = DiffMode.GPSR
    observable = Z(0) ^ circuit.n_qubits
    model = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    inputs = rand_featureparameters(circuit, 1)
    kwargs = {"n_shots": 100000}
    estimated_values = model.expectation(
        inputs,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    pyqtorch_backend = backend_factory(backend=backend, diff_mode=diff_mode)
    (conv_circ, conv_obs, embed, params) = pyqtorch_backend.convert(circuit, observable)
    pyqtorch_expectation = pyqtorch_backend.expectation(conv_circ, conv_obs, embed(params, inputs))[
        0
    ]
    assert allclose(estimated_values, pyqtorch_expectation, atol=LOW_ACCEPTANCE)


@given(st.digital_circuits())
@settings(deadline=None)
def test_basic_list_observables_tomography_for_quantum_model(circuit: QuantumCircuit) -> None:
    observable = [Z(n) for n in range(circuit.n_qubits)]
    model = QuantumModel(
        circuit=circuit,
        observable=observable,  # type: ignore
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    inputs = rand_featureparameters(circuit, 1)
    kwargs = {"n_shots": 100000}
    estimated_values = model.expectation(
        inputs,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    pyqtorch_backend = backend_factory(BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    (conv_circ, conv_obs, embed, params) = pyqtorch_backend.convert(
        circuit, observable  # type: ignore [arg-type]
    )
    pyqtorch_expectation = pyqtorch_backend.expectation(conv_circ, conv_obs, embed(params, inputs))
    assert allclose(estimated_values, pyqtorch_expectation, atol=LOW_ACCEPTANCE)


theta1 = Parameter("theta1", trainable=False)
theta2 = Parameter("theta2", trainable=False)
theta3 = Parameter("theta3", trainable=False)
theta4 = Parameter("theta4", trainable=False)

blocks = chain(
    kron(RX(0, theta1), RY(1, theta2)),
    kron(RX(0, theta3), RY(1, theta4)),
)

values = {
    "theta1": tensor([0.5]),
    "theta2": tensor([1.5]),
    "theta3": tensor([2.0]),
    "theta4": tensor([2.5]),
}

values2 = {
    "theta1": tensor([0.5, 1.0]),
    "theta2": tensor([1.5, 2.0]),
    "theta3": tensor([2.0, 2.5]),
    "theta4": tensor([2.5, 3.0]),
}


@pytest.mark.parametrize(
    "circuit, values",
    [
        (
            QuantumCircuit(2, blocks),
            values,
        ),
        (
            QuantumCircuit(2, blocks),
            values2,
        ),
    ],
)
def test_basic_tomography_for_parametric_circuit_forward_pass(
    circuit: QuantumCircuit, values: dict
) -> None:
    observable = Z(0) ^ circuit.n_qubits
    model = QuantumModel(
        circuit=circuit,
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    kwargs = {"n_shots": 100000}
    estimated_values = model.expectation(
        values=values,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    pyqtorch_backend = backend_factory(BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    (conv_circ, conv_obs, embed, params) = pyqtorch_backend.convert(circuit, observable)
    pyqtorch_expectation = pyqtorch_backend.expectation(conv_circ, conv_obs, embed(params, values))
    assert allclose(estimated_values, pyqtorch_expectation, atol=LOW_ACCEPTANCE)


# The ising hamiltonian constructor produces results that
# are far at variance. This is investigated separately.
@pytest.mark.slow
@pytest.mark.parametrize(
    "observable, acceptance",
    [
        (total_magnetization(4), MIDDLE_ACCEPTANCE),
        (zz_hamiltonian(4), MIDDLE_ACCEPTANCE),
        # (ising_hamiltonian(4), MIDDLE_ACCEPTANCE),
        (
            add(
                0.5 * kron(X(0), X(1), X(2), X(3)),
                1.5 * kron(Y(0), Y(1), Y(2), Y(3)),
                2.0 * kron(Z(0), Z(1), Z(2), Z(3)),
            ),
            MIDDLE_ACCEPTANCE,
        ),
        (
            add(
                0.5 * kron(X(0), Y(1), X(2), Y(3)),
                1.5 * kron(Y(0), Z(1), Y(2), Z(3)),
                2.0 * kron(Z(0), X(1), Z(2), X(3)),
            ),
            MIDDLE_ACCEPTANCE,
        ),
    ],
)
def test_forward_and_backward_passes_with_qnn(observable: AbstractBlock, acceptance: float) -> None:
    n_qubits = 4
    batch_size = 5
    kwargs = {"n_shots": 1000000}

    # fm = fourier_feature_map(n_qubits)
    fm = feature_map(n_qubits, fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=2)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    values = {"phi": rand(batch_size, requires_grad=True)}

    measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs)

    model_with_psr = QNN(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    model_with_psr_and_init = QNN(
        circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR, measurement=measurement
    )
    model_with_psr.zero_grad()
    expectation_tomo = model_with_psr.expectation(
        values=values,
        measurement=measurement,
    )
    expectation_tomo_init = model_with_psr_and_init.expectation(values=values)
    assert allclose(expectation_tomo, expectation_tomo_init, atol=acceptance)
    dexpval_tomo = autograd.grad(
        expectation_tomo,
        values["phi"],
        ones_like(expectation_tomo),
    )[0]
    dexpval_tomo_init = autograd.grad(
        expectation_tomo_init,
        values["phi"],
        ones_like(expectation_tomo_init),
    )[0]
    assert allclose(dexpval_tomo, dexpval_tomo_init, atol=acceptance)
    expectation_exact = model_with_psr.expectation(values=values)
    dexpval_exact = autograd.grad(
        expectation_exact,
        values["phi"],
        ones_like(expectation_exact),
    )[0]
    assert allclose(expectation_tomo, expectation_exact, atol=acceptance)
    assert allclose(dexpval_tomo, dexpval_exact, atol=acceptance)


@pytest.mark.slow
@pytest.mark.parametrize(
    "observable, acceptance",
    [
        (total_magnetization(4), MIDDLE_ACCEPTANCE),
    ],
)
def test_partial_derivatives_with_qnn(observable: AbstractBlock, acceptance: float) -> None:
    n_qubits = 4
    batch_size = 5
    kwargs = {"n_shots": 100000}

    # fm = fourier_feature_map(n_qubits)
    fm = feature_map(n_qubits, fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=2)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    values = {"phi": rand(batch_size, requires_grad=True)}

    model_with_psr = QNN(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    params = {k: v for k, v in model_with_psr._params.items() if v.requires_grad}
    model_with_psr.zero_grad()
    expectation_tomo = model_with_psr.expectation(
        values=values,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    dexpval_tomo_phi = autograd.grad(
        expectation_tomo,
        values["phi"],
        ones_like(expectation_tomo),
        create_graph=True,
    )[0]
    dexpval_tomo_theta = autograd.grad(
        expectation_tomo,
        list(params.values()),
        ones_like(expectation_tomo),
        create_graph=True,
    )[0]
    dexpval_tomo_phitheta = autograd.grad(
        dexpval_tomo_phi,
        list(params.values()),
        ones_like(dexpval_tomo_phi),
        create_graph=True,
    )[0]
    d2expval_tomo_phi2 = autograd.grad(
        dexpval_tomo_phi,
        values["phi"],
        ones_like(dexpval_tomo_phi),
        create_graph=True,
    )[0]
    d2expval_tomo_phi2theta = autograd.grad(
        d2expval_tomo_phi2,
        list(params.values()),
        ones_like(d2expval_tomo_phi2),
        create_graph=True,
    )[0]
    expectation_exact = model_with_psr.expectation(values=values)
    dexpval_exact_phi = autograd.grad(
        expectation_exact,
        values["phi"],
        ones_like(expectation_exact),
        create_graph=True,
    )[0]
    dexpval_exact_theta = autograd.grad(
        expectation_exact,
        list(params.values()),
        ones_like(expectation_exact),
        create_graph=True,
    )[0]
    dexpval_exact_phitheta = autograd.grad(
        dexpval_exact_phi,
        list(params.values()),
        ones_like(dexpval_exact_phi),
        create_graph=True,
    )[0]
    d2expval_exact_phi2 = autograd.grad(
        dexpval_exact_phi,
        values["phi"],
        ones_like(dexpval_exact_phi),
        create_graph=True,
    )[0]
    d2expval_exact_phi2theta = autograd.grad(
        d2expval_exact_phi2,
        list(params.values()),
        ones_like(d2expval_exact_phi2),
        create_graph=True,
    )[0]
    assert allclose(expectation_tomo, expectation_exact, atol=acceptance)
    assert allclose(dexpval_tomo_phi, dexpval_exact_phi, atol=acceptance)
    assert allclose(dexpval_tomo_theta, dexpval_exact_theta, atol=acceptance)
    assert allclose(dexpval_tomo_phitheta, dexpval_exact_phitheta, atol=acceptance)
    assert allclose(d2expval_tomo_phi2, d2expval_exact_phi2, atol=HIGH_ACCEPTANCE)
    assert allclose(d2expval_tomo_phi2theta, d2expval_exact_phi2theta, atol=HIGH_ACCEPTANCE)


@pytest.mark.skip(
    reason="High-order derivatives takes a long time. Keeping them here for future reference."
)
@pytest.mark.parametrize(
    "observable, acceptance",
    [
        (total_magnetization(4), MIDDLE_ACCEPTANCE),
    ],
)
def test_high_order_derivatives_with_qnn(observable: AbstractBlock, acceptance: float) -> None:
    n_qubits = 4
    batch_size = 5
    kwargs = {"n_shots": 100000}

    # fm = fourier_feature_map(n_qubits)
    fm = feature_map(n_qubits, fm_type=BasisSet.CHEBYSHEV)
    ansatz = hea(n_qubits, depth=2)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    values = {"phi": rand(batch_size, requires_grad=True)}

    model_with_psr = QNN(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    params = {k: v for k, v in model_with_psr._params.items() if v.requires_grad}
    model_with_psr.zero_grad()
    expectation_tomo = model_with_psr.expectation(
        values=values,
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    dexpval_tomo_phi = autograd.grad(
        expectation_tomo,
        values["phi"],
        ones_like(expectation_tomo),
        create_graph=True,
    )[0]
    d2expval_tomo_phi2 = autograd.grad(
        dexpval_tomo_phi,
        values["phi"],
        ones_like(dexpval_tomo_phi),
        create_graph=True,
    )[0]
    d3expval_tomo_phi3 = autograd.grad(
        d2expval_tomo_phi2,
        values["phi"],
        ones_like(d2expval_tomo_phi2),
        create_graph=True,
    )[0]
    expectation_exact = model_with_psr.expectation(values=values)
    dexpval_exact_phi = autograd.grad(
        expectation_exact,
        values["phi"],
        ones_like(expectation_exact),
        create_graph=True,
    )[0]
    d2expval_exact_phi2 = autograd.grad(
        dexpval_exact_phi,
        values["phi"],
        ones_like(dexpval_exact_phi),
        create_graph=True,
    )[0]
    d3expval_exact_phi3 = autograd.grad(
        d2expval_exact_phi2,
        values["phi"],
        ones_like(d2expval_exact_phi2),
        create_graph=True,
    )[0]
    assert allclose(expectation_tomo, expectation_exact, atol=acceptance)
    assert allclose(dexpval_tomo_phi, dexpval_exact_phi, atol=acceptance)
    assert allclose(d2expval_tomo_phi2, d2expval_exact_phi2, atol=HIGH_ACCEPTANCE)
    assert allclose(d3expval_tomo_phi3, d3expval_exact_phi3, atol=HIGH_ACCEPTANCE)


def test_chemistry_hamiltonian() -> None:
    from qadence import load

    circuit = load("./tests/test_files/chem_circ.json")
    assert isinstance(circuit, QuantumCircuit)
    hamiltonian = load("./tests/test_files/chem_ham.json")
    assert isinstance(hamiltonian, AbstractBlock)
    model = QuantumModel(
        circuit=circuit,
        observable=hamiltonian,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    kwargs = {"n_shots": 1000000}
    exact = model.expectation(
        values={},
    )
    estim = model.expectation(
        values={},
        measurement=Measurements(protocol=Measurements.TOMOGRAPHY, options=kwargs),
    )
    assert allclose(estim, exact, atol=LOW_ACCEPTANCE)
