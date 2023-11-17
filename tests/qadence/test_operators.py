from __future__ import annotations

import numpy as np
import pytest
import torch
from openfermion import QubitOperator, get_sparse_operator
from torch.linalg import eigvals

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    add,
    block_is_commuting_hamiltonian,
    chain,
    from_openfermion,
    kron,
    to_openfermion,
)
from qadence.blocks.block_to_tensor import block_to_tensor
from qadence.operations import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    MCPHASE,
    MCRX,
    MCRY,
    MCRZ,
    RX,
    RY,
    RZ,
    SWAP,
    AnalogSWAP,
    H,
    HamEvo,
    I,
    N,
    S,
    T,
    Toffoli,
    X,
    Y,
    Z,
    Zero,
)


def hamevo_generator_tensor() -> torch.Tensor:
    n_qubits = 4
    h = torch.rand(2**n_qubits, 2**n_qubits)
    ham = h + torch.conj(torch.transpose(h, 0, 1))
    return ham


def hamevo_generator_block() -> AbstractBlock:
    n_qubits = 4
    ops = [X, Y] * 2
    qubit_supports = np.random.choice(n_qubits, len(ops), replace=True)
    ham = chain(
        add(*[op(q) for op, q in zip(ops, qubit_supports)]),
        *[op(q) for op, q in zip(ops, qubit_supports)],
    )
    ham = ham + ham.dagger()  # type: ignore [assignment]
    return ham


def hamevo_eigenvalues(p: float, generator: torch.Tensor) -> torch.Tensor:
    eigenvals = eigvals(generator).real
    return torch.exp(-1j * p * eigenvals)


def eigenval(p: float) -> torch.Tensor:
    return torch.exp(torch.tensor([-1j]) * p / 2.0)


def rxyz_eigenvals(p: float) -> torch.Tensor:
    return torch.cat((eigenval(p), eigenval(p).conj()))


def crxy_eigenvals(p: float, n_qubits: int = 2) -> torch.Tensor:
    return torch.cat((torch.ones(2**n_qubits - 2), eigenval(p), eigenval(p).conj()))


def crz_eigenvals(p: float, n_qubits: int = 2) -> torch.Tensor:
    return torch.cat((torch.ones(2**n_qubits - 2), eigenval(p), eigenval(p).conj()))


def cphase_eigenvals(p: float, n_qubits: int = 2) -> torch.Tensor:
    return torch.cat((torch.ones(2**n_qubits - 1), eigenval(2.0 * p).conj()))


@pytest.mark.parametrize(
    "gate, eigenvalues",
    [
        (X(0), (-1, 1)),
        (Y(0), (-1, 1)),
        (Z(0), (-1, 1)),
        (N(0), (0, 1)),
        (H(0), (-1, 1)),
        (I(0), (1, 1)),
        (Zero(), (0, 0)),
        (RX(0, 0.5), rxyz_eigenvals(0.5)),
        (RY(0, 0.5), rxyz_eigenvals(0.5)),
        (RZ(0, 0.5), rxyz_eigenvals(0.5)),
        (CNOT(0, 1), (-1, 1, 1, 1)),
        (Toffoli((0, 1), 2), (-1, 1, 1, 1, 1, 1, 1, 1)),
        (Toffoli((0, 1, 2), 3), (-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        (HamEvo(hamevo_generator_tensor(), 0.5, tuple(range(4))), ()),
        (HamEvo(hamevo_generator_block(), 0.5, tuple(range(4))), ()),
        (CRX(0, 1, 0.5), crxy_eigenvals(0.5)),
        (CRY(0, 1, 0.5), crxy_eigenvals(0.5)),
        (CRZ(0, 1, 0.5), crz_eigenvals(0.5)),
        (MCRX((0, 1), 2, 0.5), crxy_eigenvals(0.5, 3)),
        (MCRY((0, 1), 2, 0.5), crxy_eigenvals(0.5, 3)),
        (MCRZ((0, 1), 2, 0.5), crz_eigenvals(0.5, 3)),
        (T(0), (1, np.sqrt(1j))),
        (S(0), (1, 1j)),
        (SWAP(0, 1), (-1, 1, 1, 1)),
        (AnalogSWAP(0, 1), (-1, -1, -1, 1)),  # global phase difference with SWAP
        (CPHASE(0, 1, 0.5), cphase_eigenvals(0.5)),
        (MCPHASE((0, 1), 2, 0.5), cphase_eigenvals(0.5, 3)),
    ],
)
def test_gate_instantiation(gate: AbstractBlock, eigenvalues: torch.Tensor) -> None:
    if not isinstance(eigenvalues, torch.Tensor):
        eigenvalues = torch.tensor(eigenvalues, dtype=torch.cdouble)

    assert gate.qubit_support == tuple(range(gate.n_qubits))

    if isinstance(gate, HamEvo) and not isinstance(gate, AnalogSWAP):
        if isinstance(gate.generator, AbstractBlock):
            generator = block_to_tensor(gate.generator)
        elif isinstance(gate.generator, torch.Tensor):
            generator = gate.generator

        evs = hamevo_eigenvalues(0.5, generator)
        # cope with machine precision on the Gitlab runner instance
        assert torch.allclose(gate.eigenvalues, evs, atol=1e-9, rtol=1e-9)
    else:
        # cope with machine precision on the Gitlab runner instance
        assert torch.allclose(gate.eigenvalues, eigenvalues, atol=1e-9, rtol=1e-9)


def test_creation() -> None:
    from qadence.parameters import evaluate

    block1 = from_openfermion(0.52 * QubitOperator("X0 Y5") + QubitOperator("Z0"))
    block2 = from_openfermion(0.52 * QubitOperator("Z0 X5") + QubitOperator("Y0"))
    block3 = block1 + block2

    assert len(block1) == 2
    assert evaluate(block1.blocks[0].parameters.parameter) == 0.52  # type: ignore
    assert evaluate(block1.blocks[1].parameters.parameter) == 1.0  # type: ignore

    assert isinstance(block3, AddBlock)


def test_commutation() -> None:
    block1 = X(0)
    block2 = from_openfermion(0.52 * QubitOperator("X0 Y5") + QubitOperator("Z1"))
    block3 = from_openfermion(0.52 * QubitOperator("Z0 X5") + QubitOperator("Y0"))
    block4 = block2 + block3
    block5 = block3 + block4

    assert block_is_commuting_hamiltonian(block1)
    assert block_is_commuting_hamiltonian(block2)
    assert not block_is_commuting_hamiltonian(block3)
    assert not block_is_commuting_hamiltonian(block4)
    assert not block_is_commuting_hamiltonian(block5)


@pytest.mark.parametrize(
    "block_and_mat",
    [
        (X(0), np.array([[0.0, 1.0], [1.0, 0.0]])),
        (Y(0), np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 - 0.0j]])),
        (Z(0), np.array([[1.0, 0.0], [0.0, -1.0]])),
        (
            add(kron(X(0), X(1)), kron(Y(0), Y(1))) * 0.5,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_to_matrix(block_and_mat: tuple[AbstractBlock, np.ndarray]) -> None:
    block = block_and_mat[0]
    expected = block_and_mat[1]
    mat = block_to_tensor(block).squeeze().numpy()
    assert np.array_equal(mat, expected)


@pytest.mark.parametrize(
    "qubit_op",
    [
        # various
        # QubitOperator(""),
        QubitOperator("") + QubitOperator("X0") + QubitOperator("X0 X1"),
        QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1"),
        QubitOperator("X0 X1 X2 X3"),
        0.52 * QubitOperator("X0 Y5") + QubitOperator("Z0"),
        0.52 * QubitOperator("Z0 X5") + QubitOperator("Y0"),
        # total magnetization
        QubitOperator("Z0") + QubitOperator("Z1") + QubitOperator("Z2") + QubitOperator("Z3"),
        # ising-like
        QubitOperator("Z0 Z1")
        + QubitOperator("X0")
        + QubitOperator("X1")
        + QubitOperator("Z0")
        + QubitOperator("Z1"),
    ],
)
def test_from_openfermion(qubit_op: QubitOperator) -> None:
    obs = from_openfermion(qubit_op)
    expected_mat = get_sparse_operator(qubit_op).toarray()
    np_mat = block_to_tensor(obs).squeeze().numpy()
    assert np.array_equal(np_mat, expected_mat)
    op = to_openfermion(obs)
    assert op == qubit_op
