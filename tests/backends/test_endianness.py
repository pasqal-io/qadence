from __future__ import annotations

from typing import Counter

import pytest
import strategies as st  # type: ignore
import torch
from hypothesis import given, settings
from metrics import ATOL_DICT, JS_ACCEPTANCE  # type: ignore
from torch import Tensor, allclose, pi, tensor

from qadence import QuantumCircuit, block_to_tensor, run, sample
from qadence.backend import BackendName
from qadence.backends.api import backend_factory
from qadence.blocks import AbstractBlock, MatrixBlock, chain, kron
from qadence.divergences import js_divergence
from qadence.ml_tools.utils import rand_featureparameters
from qadence.operations import CNOT, RX, RY, H, HamEvo, I, X, Z
from qadence.register import Register
from qadence.states import equivalent_state, product_state
from qadence.transpile import invert_endianness
from qadence.types import Endianness, ResultType
from qadence.utils import (
    basis_to_int,
    nqubits_to_basis,
)

BACKENDS = BackendName.list()
BACKENDS.remove(BackendName.PULSER)
N_SHOTS = 1000


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "block,n_qubits,samples",
    [
        (I(0) @ X(1) @ I(2) @ X(3), 4, [Counter({"0101": N_SHOTS})]),
        (
            chain(chain(chain(H(0), X(1), CNOT(0, 1)), CNOT(0, 2)), chain(CNOT(1, 3), CNOT(1, 2))),
            4,
            [Counter({"0111": N_SHOTS / 2, "1010": N_SHOTS / 2})],
        ),
    ],
)
def test_endianness_equal_sample(
    block: AbstractBlock, n_qubits: int, samples: list[Counter[str]], backend: BackendName
) -> None:
    for endianness in Endianness:
        if endianness == Endianness.LITTLE:
            samples = [invert_endianness(samples[0])]
        circ = QuantumCircuit(n_qubits, block)
        circ_samples = sample(circ, {}, backend=backend, n_shots=N_SHOTS, endianness=endianness)
        for circ_sample, smple in zip(circ_samples, samples):
            assert js_divergence(circ_sample, smple) < JS_ACCEPTANCE + ATOL_DICT[backend]


@pytest.mark.parametrize("backend", [BackendName.PYQTORCH])
def test_endianness_hamevo(backend: BackendName) -> None:
    n_qubits = 2
    gen = -0.5 * kron(I(0) - Z(0), I(1) - X(1))
    evo = HamEvo(gen, tensor([pi / 2]))
    circ = QuantumCircuit(n_qubits, evo)
    cnotgate = CNOT(0, 1)
    qc_cnot = QuantumCircuit(2, cnotgate)
    bkd = backend_factory(backend=backend)
    conv_cnot = bkd.convert(qc_cnot)
    state_10 = product_state("10")
    conv = bkd.convert(circ)
    wf_cnot = bkd.run(
        conv_cnot.circuit, conv_cnot.embedding_fn(conv_cnot.params, {}), state=state_10
    )
    wf_hamevo = bkd.run(conv.circuit, conv.embedding_fn(conv.params, {}), state=state_10)
    assert allclose(wf_cnot, wf_hamevo)
    # The first qubit is 1 and we do CNOT(0,1), so we expect "11"
    expected_samples = [Counter({"11": N_SHOTS})]
    hamevo_samples = bkd.sample(
        conv.circuit, conv.embedding_fn(conv.params, {}), n_shots=N_SHOTS, state=state_10
    )
    cnot_samples = bkd.sample(
        conv_cnot.circuit,
        conv_cnot.embedding_fn(conv_cnot.params, {}),
        n_shots=N_SHOTS,
        state=state_10,
    )

    for hamevo_sample, expected_sample in zip(hamevo_samples, expected_samples):
        assert js_divergence(hamevo_sample, expected_sample) < JS_ACCEPTANCE + ATOL_DICT[backend]

    for cnot_sample, expected_sample in zip(cnot_samples, expected_samples):
        assert js_divergence(cnot_sample, expected_sample) < JS_ACCEPTANCE + ATOL_DICT[backend]


def test_state_endianness() -> None:
    big_endian = nqubits_to_basis(2, ResultType.STRING, Endianness.BIG)
    assert big_endian[1] == "01"

    little_endian = nqubits_to_basis(2, ResultType.STRING, Endianness.LITTLE)
    assert little_endian[1] == "10"

    state_01 = tensor([[0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]])
    assert allclose(product_state(big_endian[1]), state_01)

    assert allclose(run(I(0) @ I(1), state=state_01), state_01)

    assert basis_to_int("01", Endianness.BIG) == 1


@pytest.mark.parametrize(
    "circ, truth",
    [
        (QuantumCircuit(3), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.cdouble)),
        (QuantumCircuit(3, X(0)), torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.cdouble)),
        (QuantumCircuit(3, X(1)), torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.cdouble)),
        (
            QuantumCircuit(3, chain(X(0), X(1))),
            torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.cdouble),
        ),
        (QuantumCircuit(3, X(2)), torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.cdouble)),
        (
            QuantumCircuit(3, chain(X(0), X(2))),
            torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0]], dtype=torch.cdouble),
        ),
        (
            QuantumCircuit(3, chain(X(1), X(2))),
            torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.cdouble),
        ),
        (
            QuantumCircuit(3, chain(X(0), X(1), X(2))),
            torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.cdouble),
        ),
        (QuantumCircuit(2, RY(0, torch.pi)), torch.tensor([[0, 0, 1, 0]], dtype=torch.cdouble)),
        (QuantumCircuit(2, RY(1, torch.pi)), torch.tensor([[0, 1, 0, 0]], dtype=torch.cdouble)),
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_wf_endianness(circ: QuantumCircuit, truth: Tensor, backend: BackendName) -> None:
    for endianness in Endianness:
        wf = run(circ, {}, backend=backend, endianness=endianness)
        if endianness == Endianness.LITTLE:
            truth = invert_endianness(truth)
        assert equivalent_state(wf, truth, atol=ATOL_DICT[backend])


@pytest.mark.parametrize(
    "circ, truth",
    [
        (QuantumCircuit(3, RX(0, torch.pi)), Counter({"100": 100})),
        (QuantumCircuit(3, RX(1, torch.pi)), Counter({"010": 100})),
        (QuantumCircuit(3, RX(2, torch.pi)), Counter({"001": 100})),
    ],
)
@pytest.mark.parametrize("backend", BACKENDS + [BackendName.PULSER])
def test_backend_sample_endianness(
    circ: QuantumCircuit, truth: Counter, backend: BackendName
) -> None:
    for endianness in Endianness:
        smple = sample(circ, {}, backend=backend, n_shots=100, endianness=endianness)[0]
        if endianness == Endianness.LITTLE:
            truth = invert_endianness(truth)
        assert smple == truth


@pytest.mark.parametrize(
    "init_state", [product_state("000"), product_state("010"), product_state("111")]
)
@pytest.mark.parametrize(
    "block",
    [
        RX(0, torch.pi),
        RX(1, torch.pi),
        RX(2, torch.pi),
    ],
)
def test_pulser_run_endianness(
    init_state: Tensor,
    block: AbstractBlock,
) -> None:
    register = Register.from_coordinates([(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)])
    circ = QuantumCircuit(register, block)
    for endianness in Endianness:
        wf_pyq = run(
            circ, {}, backend=BackendName.PYQTORCH, endianness=endianness, state=init_state
        )
        wf_pulser = run(
            circ, {}, backend=BackendName.PULSER, endianness=endianness, state=init_state
        )
        assert equivalent_state(wf_pyq, wf_pulser, atol=ATOL_DICT[BackendName.PULSER])


@pytest.mark.parametrize(
    "block,n_qubits,expected_mat, expected_samples",
    [
        (
            X(0),
            2,
            torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
            [Counter({"10": N_SHOTS})],
        ),
        (
            X(1),
            2,
            torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            [Counter({"01": N_SHOTS})],
        ),
    ],
)
def test_matrix_endianness(
    block: X, n_qubits: int, expected_mat: Tensor, expected_samples: list[Counter]
) -> None:
    mat = block_to_tensor(block, {}, block.qubit_support)

    matblock = MatrixBlock(mat, block.qubit_support)
    samples = sample(n_qubits, matblock, n_shots=N_SHOTS)

    assert torch.allclose(
        block_to_tensor(block, {}, tuple([i for i in range(n_qubits)])).squeeze(0).to(dtype=int),
        expected_mat,
    )
    for smple, expected_sample in zip(samples, expected_samples):
        assert js_divergence(smple, expected_sample) < JS_ACCEPTANCE


@pytest.mark.parametrize(
    "endianness,expected_mat",
    [
        (
            Endianness.BIG,
            torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
        ),
        (
            Endianness.LITTLE,
            torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        ),
    ],
)
def test_block_to_tensor_endianness(
    endianness: Endianness,
    expected_mat: Tensor,
) -> None:
    block = X(0)
    n_qubits = 2
    assert torch.allclose(
        block_to_tensor(
            block=block,
            values={},
            qubit_support=tuple([i for i in range(n_qubits)]),
            endianness=endianness,
        )
        .squeeze(0)
        .to(dtype=int),
        expected_mat,
    )


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_inversion_for_random_circuit(backend: str, circuit: QuantumCircuit) -> None:
    bknd = backend_factory(backend=backend)
    (circ, _, embed, params) = bknd.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    for endianness in Endianness:
        samples = bknd.sample(circ, embed(params, inputs), n_shots=100, endianness=endianness)
        for _sample in samples:
            double_inv_wf = invert_endianness(invert_endianness(_sample))
            assert js_divergence(double_inv_wf, _sample) < JS_ACCEPTANCE


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_wf_inversion_for_random_circuit(backend: str, circuit: QuantumCircuit) -> None:
    bknd = backend_factory(backend=backend)
    (circ, _, embed, params) = bknd.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    for endianness in Endianness:
        wf = bknd.run(circ, embed(params, inputs), endianness=endianness)
        double_inv_wf = invert_endianness(invert_endianness(wf))
        assert equivalent_state(double_inv_wf, wf)
