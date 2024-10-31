# FIXME: not all tests pass ATOL_32 (1e-7)

from __future__ import annotations

import numpy as np
import pytest
import strategies as st
import torch
from hypothesis import given, settings
from metrics import ATOL_32, ATOL_64, ATOL_E6
from torch import Tensor

from qadence import Parameter, QuantumCircuit, VariationalParameter, run
from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ParametricBlock,
    ParametricControlBlock,
    embedding,
)
from qadence.blocks.block_to_tensor import (
    IMAT,
    ZMAT,
    TensorType,
    _block_to_tensor_embedded,
    block_to_tensor,
)
from qadence.blocks.utils import add, chain, kron
from qadence.constructors import (
    feature_map,
    hamiltonian_factory,
    hea,
    ising_hamiltonian,
    qft,
    total_magnetization,
    zz_hamiltonian,
)
from qadence.operations import (
    CNOT,
    CSWAP,
    MCPHASE,
    MCRX,
    MCRY,
    MCRZ,
    MCZ,
    PHASE,
    RX,
    RY,
    RZ,
    SWAP,
    H,
    HamEvo,
    I,
    N,
    Projector,
    S,
    T,
    Toffoli,
    U,
    X,
    Y,
    Z,
)
from qadence.states import equivalent_state, random_state, zero_state
from qadence.types import BasisSet, Interaction, ReuploadScaling


def _calc_mat_vec_wavefunction(
    block: AbstractBlock, n_qubits: int, init_state: torch.Tensor, values: dict | None = None
) -> torch.Tensor:
    mat = block_to_tensor(block, values or dict(), tuple(range(n_qubits)))
    return torch.einsum("bij,kj->bi", mat, init_state)


@pytest.mark.parametrize(
    "projector, exp_projector_mat",
    [
        (
            Projector(bra="1", ket="1", qubit_support=0),
            torch.tensor([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]),
        ),
        (
            Projector(bra="10", ket="01", qubit_support=(1, 2)),
            torch.tensor(
                [
                    [
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ]
                ]
            ),
        ),
        (
            N(0),
            (IMAT - ZMAT) / 2.0,
        ),
        (
            CNOT(0, 1),
            torch.tensor(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    ]
                ]
            ),
        ),
        (
            MCZ((0, 1), 2),
            torch.tensor(
                [
                    [
                        [
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            -1.0 + 0.0j,
                        ],
                    ]
                ]
            ),
        ),
        (
            MCRX((0, 1), 2, 1.0),
            torch.tensor(
                [
                    [
                        [
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            1.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.8776 + 0.0000j,
                            0.0000 - 0.4794j,
                        ],
                        [
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 + 0.0000j,
                            0.0000 - 0.4794j,
                            0.8776 + 0.0000j,
                        ],
                    ]
                ]
            ),
        ),
        (
            CSWAP(0, 1, 2),
            torch.tensor(
                [
                    [
                        [
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                        ],
                    ]
                ]
            ),
        ),
        (
            Toffoli((0, 1), 2),
            torch.tensor(
                [
                    [
                        [
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ]
                ]
            ),
        ),
    ],
)
def test_projector_tensor(projector: AbstractBlock, exp_projector_mat: Tensor) -> None:
    projector_mat = block_to_tensor(projector)
    assert torch.allclose(projector_mat, exp_projector_mat, atol=1.0e-4)


projector0 = Projector(ket="0", bra="0", qubit_support=0)
projector1 = Projector(ket="1", bra="1", qubit_support=0)


cnot = kron(projector0, I(1)) + kron(projector1, X(1))

projector00 = Projector(ket="00", bra="00", qubit_support=(0, 1))
projector01 = Projector(ket="01", bra="10", qubit_support=(0, 1))
projector10 = Projector(ket="10", bra="01", qubit_support=(0, 1))
projector11 = Projector(ket="11", bra="11", qubit_support=(0, 1))

swap = projector00 + projector10 + projector01 + projector11


@pytest.mark.parametrize(
    "projector, exp_projector",
    [
        (
            cnot,
            CNOT(0, 1),
        ),
        (swap, SWAP(0, 1)),
    ],
)
def test_projector_composition_unitaries(
    projector: AbstractBlock, exp_projector: AbstractBlock
) -> None:
    assert torch.allclose(block_to_tensor(projector), block_to_tensor(exp_projector), atol=ATOL_64)


@given(st.batched_digital_circuits())
@settings(deadline=None)
def test_embedded(circ_and_inputs: tuple[QuantumCircuit, dict[str, torch.Tensor]]) -> None:
    circ, inputs = circ_and_inputs
    ps, embed = embedding(circ.block, to_gate_params=False)
    m = block_to_tensor(circ.block, inputs)
    m_embedded = _block_to_tensor_embedded(circ.block, values=embed(ps, inputs))
    zro_state = zero_state(circ.n_qubits)
    wf_run = run(circ, values=inputs)
    wf_embedded = torch.einsum("bij,kj->bi", m_embedded, zro_state)
    wf_nonembedded = torch.einsum("bij,kj->bi", m, zro_state)
    assert torch.allclose(m, m_embedded)
    assert equivalent_state(wf_run, wf_embedded, atol=ATOL_E6)
    assert equivalent_state(wf_run, wf_nonembedded, atol=ATOL_E6)


@pytest.mark.parametrize("n_qubits", [3, 5, 7])
@pytest.mark.parametrize("op0", [X, Y, Z])
@pytest.mark.parametrize("op1", [X, Y, Z])
def test_block_to_tensor_support(n_qubits: int, op0: X | Y | Z, op1: X | Y | Z) -> None:
    mat0 = block_to_tensor(op0(0))  # type: ignore [operator]
    mat1 = block_to_tensor(op1(0))  # type: ignore [operator]
    IMAT = block_to_tensor(I(0))

    possible_targets = list(range(n_qubits - 1))
    target = np.random.choice(possible_targets)

    qubit_support = [target, n_qubits - 1]
    np.random.shuffle(qubit_support)

    block = kron(op0(qubit_support[0]), op1(qubit_support[1]))  # type: ignore [operator]

    mat_small = block_to_tensor(block, use_full_support=False)
    mat_large = block_to_tensor(block, use_full_support=True)

    if qubit_support[0] < qubit_support[1]:
        exact_small = torch.kron(mat0, mat1).unsqueeze(0)
    else:
        exact_small = torch.kron(mat1, mat0).unsqueeze(0)

    kron_list = [IMAT for i in range(n_qubits)]
    kron_list[qubit_support[0]] = mat0
    kron_list[qubit_support[1]] = mat1

    exact_large = kron_list[0]
    for i in range(n_qubits - 1):
        exact_large = torch.kron(exact_large, kron_list[i + 1])

    assert torch.allclose(mat_small, exact_small)
    assert torch.allclose(mat_large, exact_large)


@pytest.mark.parametrize("gate", [I, X, Y, Z, H, T, S])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_single_qubit_gates(gate: AbstractBlock, n_qubits: int) -> None:
    target = np.random.randint(0, n_qubits)
    block = gate(target)  # type: ignore[operator]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("batch_size", [i for i in range(2, 10)])
@pytest.mark.parametrize("gate", [RX, RY, RZ, U])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_rotation_gates(batch_size: int, gate: ParametricBlock, n_qubits: int) -> None:
    param_names = [f"th{i}" for i in range(gate.num_parameters())]

    target = np.random.randint(0, n_qubits)
    block = gate(target, *param_names)  # type: ignore[operator]
    init_state = random_state(n_qubits)
    values = {k: torch.rand(batch_size) for k in param_names}
    wf_pyq = run(n_qubits, block, values=values, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state, values=values)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)

    # test with fixed parameter
    block = gate(target, *[np.random.rand()] * len(param_names))  # type: ignore[operator]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)

    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("gate", [MCRX, MCRY, MCRZ, MCPHASE])
@pytest.mark.parametrize("n_qubits", [2, 4, 6])
def test_controlled_parameterized_gates(gate: ParametricControlBlock, n_qubits: int) -> None:
    qubits = np.random.choice(n_qubits, size=n_qubits, replace=False).tolist()
    control = tuple(qubits[:-1])
    target = qubits[-1]
    q = np.random.choice([*control, target])
    block = chain(X(q), gate(control, target, "theta"))  # type: ignore[operator]
    values = {"theta": torch.rand(3)}
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, values=values, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state, values=values)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("gate", [CNOT, SWAP])
@pytest.mark.parametrize("n_qubits", [2, 4, 6])
def test_swap_cnot_gates(gate: AbstractBlock, n_qubits: int) -> None:
    control, target = np.random.choice(n_qubits, size=2, replace=False).tolist()
    q = np.random.choice([control, target])
    block = chain(X(q), gate(control, target))  # type: ignore[operator]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


def test_cnot_support() -> None:
    mat_full = block_to_tensor(CNOT(0, 2), use_full_support=True)
    mat_small = block_to_tensor(CNOT(0, 2), use_full_support=False)
    assert len(mat_full.squeeze(0)) == 8
    assert len(mat_small.squeeze(0)) == 4


@pytest.mark.parametrize("n_qubits", [3, 4, 6])
def test_cswap_gate(n_qubits: int) -> None:
    control, target1, target2 = np.random.choice(
        list(range(n_qubits)), size=3, replace=False
    ).tolist()
    block = CSWAP(control, target1, target2)
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [3, 4, 6])
def test_toffoli_gates(n_qubits: int) -> None:
    init_state = random_state(n_qubits)
    target = np.random.choice(n_qubits, size=1, replace=False)[0]
    control = tuple([qubit for qubit in range(n_qubits) if qubit != target])
    block = Toffoli(control, target)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)

    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
@pytest.mark.parametrize("generator_type", ["tensor", "block"])
def test_hamevo_gate(n_qubits: int, generator_type: str) -> None:
    dim = np.random.randint(1, n_qubits + 1)
    if generator_type == "tensor":
        h = torch.rand(2**dim, 2**dim)
        generator = h + torch.conj(torch.transpose(h, 0, 1))
        generator = generator.unsqueeze(0)
    elif generator_type == "block":
        ops = [X, Y] * 2
        qubit_supports = np.random.choice(dim, len(ops), replace=True)
        generator = chain(
            add(*[op(q) for op, q in zip(ops, qubit_supports)]),
            *[op(q) for op, q in zip(ops, qubit_supports)],
        )
        generator = generator + generator.dagger()

    x = Parameter("x", trainable=False)
    qubit_support = list(range(dim))
    # FIXME: random shuffle temporarily commented due to issues with how
    # qubit_support and block_to_tensor handle MatrixBlocks, to be fixed.
    # np.random.shuffle(qubit_support)
    block = HamEvo(generator, x, qubit_support=tuple(qubit_support))
    values = {"x": torch.tensor(0.5)}
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, values=values, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state, values)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", list(range(2, 9, 2)))
def test_hea(n_qubits: int, depth: int = 1) -> None:
    block = hea(n_qubits, depth)
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_total_magnetization(n_qubits: int) -> None:
    block = total_magnetization(n_qubits)
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert torch.allclose(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [1, 2, 4])
@pytest.mark.parametrize("fm_type", [BasisSet.FOURIER, BasisSet.CHEBYSHEV])
@pytest.mark.parametrize(
    "reupload_scaling", [ReuploadScaling.CONSTANT, ReuploadScaling.TOWER, ReuploadScaling.EXP]
)
@pytest.mark.parametrize("op", [RX, RY, RZ, PHASE])
def test_feature_maps(
    n_qubits: int,
    fm_type: BasisSet,
    reupload_scaling: ReuploadScaling,
    op: type[RX] | type[RY] | type[RZ] | type[PHASE],
) -> None:
    x = Parameter("x", trainable=False)
    values = {"x": torch.rand(1)}
    block = feature_map(
        n_qubits, param=x, op=op, fm_type=fm_type, reupload_scaling=reupload_scaling
    )  # type: ignore[arg-type]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state, values=values)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state, values=values)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [2, 4])
@pytest.mark.parametrize("interaction", [Interaction.ZZ, Interaction.NN, Interaction.XY])
@pytest.mark.parametrize("detuning", [Z, N, X])
def test_hamiltonians(
    n_qubits: int, interaction: Interaction, detuning: type[N] | type[X] | type[Y] | type[Z]
) -> None:
    block = hamiltonian_factory(
        n_qubits,
        interaction=interaction,
        detuning=detuning,
        random_strength=True,
    )
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert torch.allclose(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_qft_block(n_qubits: int) -> None:
    block = qft(n_qubits)
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32 * 10)


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
def test_random_qubit_support(n_qubits: int) -> None:
    dim = np.random.randint(1, n_qubits + 1)
    ops = [X, Y, Z, S, T] * 2
    qubit_supports = np.random.choice(dim, len(ops), replace=True)
    block = chain(
        *[op(q) for op, q in zip(ops, qubit_supports)],  # type: ignore [abstract]
    )
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


def variational_ising(n_qubits: int) -> AddBlock:
    ops = []
    for i in range(n_qubits):
        for j in range(i):
            x = VariationalParameter(f"x_{i}{j}")
            ops.append(x * kron(Z(j), Z(i)))
    return add(*ops)


@pytest.mark.parametrize(
    "block, is_diag_pauli",
    [
        (ising_hamiltonian(2), False),
        (total_magnetization(2), True),
        (zz_hamiltonian(2), True),
        (variational_ising(3), True),
        (hea(4), False),
    ],
)
def test_block_is_diag(block: AbstractBlock, is_diag_pauli: bool) -> None:
    assert block._is_diag_pauli == is_diag_pauli


@pytest.mark.parametrize("n_qubits", [i for i in range(1, 5)])
@pytest.mark.parametrize("obs", [total_magnetization, zz_hamiltonian])
def test_sparse_obs_conversion(n_qubits: int, obs: AbstractBlock) -> None:
    obs = obs(n_qubits)  # type: ignore[operator]
    sparse_diag = block_to_tensor(obs, tensor_type=TensorType.SPARSEDIAGONAL)
    true_diag = torch.diag(block_to_tensor(obs, {}, tuple([i for i in range(n_qubits)])).squeeze(0))

    assert torch.allclose(
        sparse_diag.coalesce().values(), true_diag.to_sparse().coalesce().values()
    )
    assert torch.allclose(
        sparse_diag.coalesce().indices(), true_diag.to_sparse().coalesce().indices()
    )


def test_scaled_kron_hamevo_equal() -> None:
    block = kron(I(0), I(1))
    assert torch.allclose(
        block_to_tensor(HamEvo(block, 0.0)), block_to_tensor(HamEvo(1.0 * block, 0.0))
    )
