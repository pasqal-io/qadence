from __future__ import annotations

from timeit import timeit

import numpy as np
import pytest
import torch
from metrics import LOW_ACCEPTANCE

from qadence.backends.api import backend_factory
from qadence.blocks import chain, kron, tag
from qadence.blocks.primitive import PrimitiveBlock
from qadence.circuit import QuantumCircuit
from qadence.operations import RX, RY, H, I, S, T, Z
from qadence.overlap import Overlap, OverlapMethod
from qadence.parameters import FeatureParameter, VariationalParameter
from qadence.types import PI, BackendName, DiffMode

torch.manual_seed(42)


def _create_test_circuits(n_qubits: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    # prepare circuit for bras
    param_bra = FeatureParameter("phi")
    block_bra = kron(*[RX(qubit, param_bra) for qubit in range(n_qubits)])
    fm_bra = tag(block_bra, tag="feature-map-bra")
    circuit_bra = QuantumCircuit(n_qubits, fm_bra)

    # prepare circuit for kets
    param_ket = FeatureParameter("psi")
    block_ket = kron(*[RX(qubit, param_ket) for qubit in range(n_qubits)])
    fm_ket = tag(block_ket, tag="feature-map-ket")
    circuit_ket = QuantumCircuit(n_qubits, fm_ket)

    return circuit_bra, circuit_ket


def _get_theoretical_result(n_qubits: int, values_bra: dict, values_ket: dict) -> torch.Tensor:
    # get theoretical result
    ovrlp_theor = torch.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ovrlp_theor[i, j] = np.cos((values_bra["phi"][i] - values_ket["psi"][j]) / 2) ** (
                2 * n_qubits
            )
    return ovrlp_theor


def _generate_parameter_values() -> tuple[dict, dict]:
    values_bra = {"phi": 2 * PI * torch.rand(2)}
    values_ket = {"psi": 2 * PI * torch.rand(2)}
    return values_bra, values_ket


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.BRAKET])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_exact(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.EXACT,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_exact = ovrlp(values_bra, values_ket)

    assert torch.all(torch.isclose(ovrlp_exact, ovrlp_theor, atol=LOW_ACCEPTANCE))


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.BRAKET])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_jensen_shannon(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra = {"phi": torch.Tensor([PI / 2, PI])}
    values_ket = {"psi": torch.Tensor([PI / 2, PI])}

    # get theoretical result
    if n_qubits == 1:
        ovrlp_theor = torch.tensor([[1.0, 0.78], [0.78, 1.0]])
    elif n_qubits == 2:
        ovrlp_theor = torch.tensor([[1.0, 0.61], [0.61, 1.0]])

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.JENSEN_SHANNON,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_js = ovrlp(values_bra, values_ket, n_shots=10000)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_js, atol=LOW_ACCEPTANCE))


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.BRAKET])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_comp_uncomp_exact(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.COMPUTE_UNCOMPUTE,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_cu = ovrlp(values_bra, values_ket)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_cu, atol=LOW_ACCEPTANCE))


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.BRAKET])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_comp_uncomp_shots(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.COMPUTE_UNCOMPUTE,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_cu = ovrlp(values_bra, values_ket, n_shots=10000)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_cu, atol=LOW_ACCEPTANCE))


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_swap_test_exact(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.SWAP_TEST,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_st = ovrlp(values_bra, values_ket)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_st, atol=LOW_ACCEPTANCE))


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_swap_test_shots(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.SWAP_TEST,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_st = ovrlp(values_bra, values_ket, n_shots=10000)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_st, atol=LOW_ACCEPTANCE))


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_hadamard_test_exact(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.HADAMARD_TEST,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_ht = ovrlp(values_bra, values_ket)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_ht, atol=LOW_ACCEPTANCE))


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH])
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_overlap_hadamard_test_shots(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuits
    circuit_bra, circuit_ket = _create_test_circuits(n_qubits)

    # values for circuits
    values_bra, values_ket = _generate_parameter_values()

    # get theoretical result
    ovrlp_theor = _get_theoretical_result(n_qubits, values_bra, values_ket)

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_ket,
        backend=backend_name,
        method=OverlapMethod.HADAMARD_TEST,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    ovrlp_ht = ovrlp(values_bra, values_ket, n_shots=10000)

    assert torch.all(torch.isclose(ovrlp_theor, ovrlp_ht, atol=LOW_ACCEPTANCE))


# TODO: investigate why braket overlap.EXACT gives slower results that fails
# TODO: move the test below in the future to https://gitlab.pasqal.com/pqs/benchmarks
@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.BRAKET])
@pytest.mark.parametrize("n_qubits", [1, 2, 4, 10, 12])
def test_overlap_exact_speed(backend_name: BackendName, n_qubits: int) -> None:
    # prepare circuit for bras
    param_bra = FeatureParameter("phi")
    block_bra = kron(*[RX(qubit, param_bra) for qubit in range(n_qubits)])
    fm_bra = tag(block_bra, tag="feature-map-bra")
    circuit_bra = QuantumCircuit(n_qubits, fm_bra)

    # values for circuits
    values_bra = {"phi": torch.Tensor([PI / 2])}

    # create backend for calculating expectation value
    obs = tag(kron(*[I(i) for i in range(n_qubits)]), "observable")
    backend = backend_factory(backend=backend_name, diff_mode=None)
    (conv_circ, conv_obs, embed, params) = backend.convert(circuit_bra, obs)
    t_exp = timeit(
        lambda: backend.expectation(conv_circ, conv_obs, embed(params, values_bra)), number=100
    )

    # get result from overlap class
    ovrlp = Overlap(
        circuit_bra,
        circuit_bra,
        backend=backend_name,
        method=OverlapMethod.EXACT,
        diff_mode=DiffMode.AD if backend_name == BackendName.PYQTORCH else DiffMode.GPSR,
    )
    t_ovrlp = timeit(lambda: ovrlp(values_bra, values_bra), number=100)

    assert np.round(t_ovrlp / t_exp, decimals=0) <= 2.0


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH])
@pytest.mark.parametrize("gate", [Z, S, T, H])
def test_overlap_training(backend_name: BackendName, gate: PrimitiveBlock) -> None:
    # define training parameters
    phi = VariationalParameter("phi")
    theta = VariationalParameter("theta")

    # define training and target quantum circuits
    circuit_bra = QuantumCircuit(1, chain(RX(0, phi), RY(0, theta)))
    circuit_ket = QuantumCircuit(1, gate(0))  # type: ignore [operator]

    # define overlap model
    model = Overlap(circuit_bra, circuit_ket, backend=backend_name, method=OverlapMethod.EXACT)

    # prepare for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)
    loss_criterion = torch.nn.MSELoss()
    n_epochs = 1000
    loss_save = []

    # train model
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model()
        loss = loss_criterion(torch.tensor(1.0).reshape((1, 1)), out)
        loss.backward()
        optimizer.step()
        loss_save.append(loss.item())

    # get final results
    wf_exact = model.ket_model.run({}).detach()
    wf_overlap = model.run({}).detach()

    assert torch.all(torch.isclose(wf_exact, wf_overlap, atol=LOW_ACCEPTANCE))


def test_output_shape() -> None:
    # define feature params
    param_bra = FeatureParameter("phi")
    param_ket = FeatureParameter("psi")

    # prepare circuit for bras
    block_bra = kron(*[RX(qubit, param_bra) for qubit in range(2)])
    fm_bra = tag(block_bra, tag="feature-map-bra")
    circuit_bra = QuantumCircuit(2, fm_bra)

    # prepare circuit for kets
    block_ket = kron(*[RX(qubit, param_ket) for qubit in range(2)])
    fm_ket = tag(block_ket, tag="feature-map-ket")
    circuit_ket = QuantumCircuit(2, fm_ket)

    # values for circuits
    values_bra = {"phi": 2 * PI * torch.rand(2)}
    values_ket = {"psi": 2 * PI * torch.rand(3)}

    # get result from overlap class - distinct feature params for bra/ket
    ovrlp = Overlap(
        circuit_bra, circuit_ket, backend=BackendName.PYQTORCH, method=OverlapMethod.EXACT
    )
    ovrlp = ovrlp(values_bra, values_ket)
    assert ovrlp.shape == (2, 3)

    # prepare circuit for bras
    block_bra = kron(*[RX(qubit, param_bra) for qubit in range(2)])
    fm_bra = tag(block_bra, tag="feature-map-bra")
    circuit_bra = QuantumCircuit(2, fm_bra)

    # prepare circuit for kets
    block_ket = kron(*[RX(qubit, param_bra) for qubit in range(2)])
    fm_ket = tag(block_ket, tag="feature-map-ket")
    circuit_ket = QuantumCircuit(2, fm_ket)

    # values for circuits
    values_bra = {"phi": 2 * PI * torch.rand(4)}

    # get result from overlap class - shared feature param for bra/ket
    ovrlp = Overlap(
        circuit_bra, circuit_ket, backend=BackendName.PYQTORCH, method=OverlapMethod.EXACT
    )
    ovrlp = ovrlp(values_bra, values_bra)
    assert ovrlp.shape == (4, 1)

    # prepare circuit for kets
    block_ket = kron(*[RX(qubit, PI / 2) for qubit in range(2)])
    fm_ket = tag(block_ket, tag="feature-map-ket")
    circuit_ket = QuantumCircuit(2, fm_ket)

    # values for circuits
    values_bra = {"phi": 2 * PI * torch.rand(4)}

    # get result from overlap class - bra has feature param, ket doesn't
    ovrlp = Overlap(
        circuit_bra, circuit_ket, backend=BackendName.PYQTORCH, method=OverlapMethod.EXACT
    )
    ovrlp = ovrlp(values_bra)
    assert ovrlp.shape == (4, 1)
