from __future__ import annotations

import numpy as np
import pytest
import torch

from qadence import QuantumModel as QM
from qadence.backends.api import DiffMode
from qadence.blocks import MatrixBlock, ParametricBlock, PrimitiveBlock, chain
from qadence.blocks.block_to_tensor import OPERATIONS_DICT, block_to_tensor
from qadence.circuit import QuantumCircuit as QC
from qadence.constructors import hea
from qadence.execution import run
from qadence.operations import CNOT, RX, RY, RZ, H, I, S, T, U, X, Y, Z
from qadence.states import random_state
from qadence.types import BackendName


@pytest.mark.parametrize("gate", [I, X, Y, Z, H, T, S])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_single_qubit_gates(gate: PrimitiveBlock, n_qubits: int) -> None:
    target = np.random.randint(0, n_qubits)
    block = gate(target)  # type: ignore[operator]
    mat = block_to_tensor(block, {}, tuple(range(n_qubits)))
    matblock = MatrixBlock(block_to_tensor(block, {}, (target,)), (target,))
    init_state = random_state(n_qubits)
    wf_pyq_mat = run(n_qubits, matblock, state=init_state)
    wf_pyq_standard = run(n_qubits, block, state=init_state)
    wf_mat = torch.einsum("bij,kj->ki", mat, init_state)
    assert torch.all(torch.isclose(wf_pyq_mat, wf_mat)) and torch.all(
        torch.isclose(wf_pyq_mat, wf_pyq_standard)
    )


@pytest.mark.parametrize("gate", [RX, RY, RZ, U])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_rotation_gates(gate: ParametricBlock, n_qubits: int) -> None:
    target = np.random.randint(0, n_qubits)
    param = [np.random.rand()] * gate.num_parameters()
    block = gate(target, *param)  # type: ignore[operator]
    init_state = random_state(n_qubits)
    mat = block_to_tensor(block, {}, tuple(range(n_qubits)))
    matblock = MatrixBlock(block_to_tensor(block, {}, (target,)), (target,))
    wf_pyq_mat = run(n_qubits, matblock, state=init_state)
    wf_pyq_standard = run(n_qubits, block, state=init_state)
    wf_mat = torch.einsum("bij,kj->ki", mat, init_state)
    assert torch.allclose(wf_pyq_mat, wf_mat) and torch.allclose(wf_pyq_mat, wf_pyq_standard)


@pytest.mark.parametrize("gate", [X, Y, Z])
def test_single_qubit_gates_eigenvals(gate: PrimitiveBlock) -> None:
    matblock = MatrixBlock(OPERATIONS_DICT[gate.name], (0,))
    block = gate(0)  # type: ignore[operator]
    assert torch.allclose(matblock.eigenvalues, block.eigenvalues)


@pytest.mark.parametrize("gate", [RX, RY, RZ, U])
@pytest.mark.parametrize("n_qubits", [2, 4])
def test_parametric_circ_with_matblock(gate: ParametricBlock, n_qubits: int) -> None:
    target = np.random.randint(0, n_qubits)
    p = [np.random.rand()] * gate.num_parameters()
    block = gate(target, *p)  # type: ignore[operator]
    s = random_state(n_qubits)
    matblock = MatrixBlock(block_to_tensor(block, {}, (target,)), (target,))
    bb = chain(hea(n_qubits=n_qubits, depth=1), CNOT(0, 1))
    wf_pyq_mat = run(n_qubits, chain(matblock, bb), state=s)
    wf_pyq_standard = run(n_qubits, chain(gate(target, *p), bb), state=s)  # type: ignore[operator]
    assert torch.all(torch.isclose(wf_pyq_mat, wf_pyq_standard))


def test_qm_with_matblock() -> None:
    n_qubits = 1
    XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
    state = random_state(n_qubits)
    matblock = MatrixBlock(XMAT, (0,))

    qm_mat = QM(
        circuit=QC(n_qubits, matblock),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    qm = QM(
        circuit=QC(n_qubits, X(0)),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    wf_mat = qm_mat.run({}, state)
    exp_mat = qm_mat.expectation({})
    wf = qm.run({}, state)
    exp = qm.expectation({})

    assert torch.all(torch.isclose(wf_mat, wf)) and torch.isclose(exp, exp_mat)
