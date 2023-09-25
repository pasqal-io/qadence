# FIXME: not all tests pass ATOL_32 (1e-7)

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import strategies as st
import torch
from hypothesis import given, settings
from metrics import ATOL_32, ATOL_E6

from qadence import Parameter, QuantumCircuit, VariationalParameter, run
from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ParametricBlock,
    ParametricControlBlock,
    embedding,
)
from qadence.blocks.block_to_tensor import (
    TensorType,
    _block_to_tensor_embedded,
    block_to_tensor,
)
from qadence.blocks.utils import add, chain, kron
from qadence.constructors import (
    feature_map,
    hea,
    ising_hamiltonian,
    nn_hamiltonian,
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
    RX,
    RY,
    RZ,
    SWAP,
    H,
    HamEvo,
    I,
    S,
    T,
    Toffoli,
    U,
    X,
    Y,
    Z,
)
from qadence.states import equivalent_state, random_state, zero_state


def _calc_mat_vec_wavefunction(
    block: AbstractBlock, n_qubits: int, init_state: torch.Tensor, values: dict = {}
) -> torch.Tensor:
    mat = block_to_tensor(block, values, tuple(range(n_qubits)))
    return torch.einsum("bij,kj->bi", mat, init_state)


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
    qubits = np.random.choice(list(range(n_qubits)), size=n_qubits, replace=False).tolist()
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
    control, target = np.random.choice(list(range(n_qubits)), size=2, replace=False).tolist()
    q = np.random.choice([control, target])
    block = chain(X(q), gate(control, target))  # type: ignore[operator]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


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
    target = np.random.choice(list(range(n_qubits)), size=1, replace=False)[0]
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
        qubit_supports = np.random.choice(list(range(dim)), len(ops), replace=True)
        generator = chain(
            add(*[op(q) for op, q in zip(ops, qubit_supports)]),
            *[op(q) for op, q in zip(ops, qubit_supports)],
        )
        generator = generator + generator.dagger()

    x = Parameter("x", trainable=False)
    qubit_support = list(range(dim))
    np.random.shuffle(qubit_support)
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
@pytest.mark.parametrize("fm_type", ["tower", "fourier", "chebyshev"])
@pytest.mark.parametrize("op", [RX, RY, RZ])
def test_feature_maps(n_qubits: int, fm_type: str, op: AbstractBlock) -> None:
    x = Parameter("x", trainable=True)
    block = feature_map(n_qubits, param=x, op=op, fm_type=fm_type)  # type: ignore[arg-type]
    init_state = random_state(n_qubits)
    wf_pyq = run(n_qubits, block, state=init_state)
    wf_mat = _calc_mat_vec_wavefunction(block, n_qubits, init_state)
    assert equivalent_state(wf_pyq, wf_mat, atol=ATOL_32)


@pytest.mark.parametrize("n_qubits", [2, 4])
@pytest.mark.parametrize("ham_constructor", [ising_hamiltonian, zz_hamiltonian, nn_hamiltonian])
def test_hamiltonians(n_qubits: int, ham_constructor: Callable) -> None:
    block = ham_constructor(n_qubits)
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
    qubit_supports = np.random.choice(list(range(dim)), len(ops), replace=True)
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
