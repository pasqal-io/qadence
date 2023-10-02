from __future__ import annotations

import os
from json import loads
from typing import no_type_check

import numpy as np
import pytest
import torch
from metrics import ATOL_32, DIGITAL_DECOMP_ACCEPTANCE_HIGH, DIGITAL_DECOMP_ACCEPTANCE_LOW

from qadence import BackendName, DiffMode
from qadence.blocks import (
    AbstractBlock,
    add,
    chain,
    get_pauli_blocks,
    kron,
    primitive_blocks,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors import (
    ising_hamiltonian,
    total_magnetization,
    zz_hamiltonian,
)
from qadence.models import QuantumModel
from qadence.operations import (
    CNOT,
    RX,
    RZ,
    H,
    HamEvo,
    X,
    Y,
    Z,
)
from qadence.parameters import Parameter, VariationalParameter, evaluate
from qadence.serialization import deserialize
from qadence.types import LTSOrder


@no_type_check
def test_hamevo_digital_decompositon() -> None:
    parameter = Parameter("p", trainable=True)

    # simple Pauli
    generator = Z(0)
    expected = chain(RZ(0, parameter=parameter))
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    for exp, blk in zip(primitive_blocks(expected), primitive_blocks(tevo_digital)):
        assert type(exp) == type(blk)
        assert exp.qubit_support == blk.qubit_support

    # commuting
    generator = add(kron(Y(0), Y(1)))
    expected = chain(
        RX(0, parameter=1.5708),
        RX(1, parameter=1.5708),
        CNOT(0, 1),
        RZ(1, parameter=parameter),
        CNOT(0, 1),
        RX(0, parameter=-1.5708),
        RX(1, parameter=-1.5708),
    )
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    for exp, blk in zip(primitive_blocks(expected), primitive_blocks(tevo_digital)):
        assert type(exp) == type(blk)
        if isinstance(blk, RX):
            exp_p, blk_p = exp.parameters.parameter, blk.parameters.parameter
            assert np.isclose(evaluate(exp_p), evaluate(blk_p))

    # Trotter
    generator = kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2))
    expected = chain(
        CNOT(0, 1),
        CNOT(1, 2),
        RZ(2, parameter=parameter),
        CNOT(1, 2),
        CNOT(0, 1),
        H(0),
        RX(1, parameter=-1.5708),
        CNOT(0, 1),
        CNOT(1, 2),
        RZ(2, parameter=parameter),
        CNOT(1, 2),
        CNOT(0, 1),
        H(0),
        RX(1, parameter=-1.5708),
    )
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    assert all(primitive_blocks(expected)) == all(primitive_blocks(tevo_digital))


@no_type_check
def test_hamevo_digital_decompositon_multiparam_timeevo() -> None:
    p0 = Parameter("p0", trainable=True)
    p1 = Parameter("p1", trainable=True)

    parameter = p0 + p1

    # simple Pauli
    generator = Z(0)
    expected = chain(RZ(0, parameter=parameter))
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    for exp, blk in zip(primitive_blocks(expected), primitive_blocks(tevo_digital)):
        assert type(exp) == type(blk)
        assert exp.qubit_support == blk.qubit_support

    # commuting
    generator = add(kron(Y(0), Y(1)))
    expected = chain(
        RX(0, parameter=1.5708),
        RX(1, parameter=1.5708),
        CNOT(0, 1),
        RZ(1, parameter=parameter),
        CNOT(0, 1),
        RX(0, parameter=-1.5708),
        RX(1, parameter=-1.5708),
    )
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    for exp, blk in zip(primitive_blocks(expected), primitive_blocks(tevo_digital)):
        assert type(exp) == type(blk)
        if isinstance(blk, RX):
            exp_p, blk_p = exp.parameters.parameter, blk.parameters.parameter
            assert np.isclose(evaluate(exp_p), evaluate(blk_p))

    # Trotter
    generator = kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2))
    expected = chain(
        CNOT(0, 1),
        CNOT(1, 2),
        RZ(2, parameter=parameter),
        CNOT(1, 2),
        CNOT(0, 1),
        H(0),
        RX(1, parameter=-1.5708),
        CNOT(0, 1),
        CNOT(1, 2),
        RZ(2, parameter=parameter),
        CNOT(1, 2),
        CNOT(0, 1),
        H(0),
        RX(1, parameter=-1.5708),
    )
    tevo_digital = HamEvo(generator, parameter).digital_decomposition(approximation=LTSOrder.BASIC)
    assert all(primitive_blocks(expected)) == all(primitive_blocks(tevo_digital))


@pytest.mark.parametrize(
    "generator",
    [
        X(0),
        Y(0),
        Z(0),
        kron(X(0), X(1)),
        kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
        add(Z(0), Z(1), Z(2)),
        0.1 * kron(X(0), X(1)) + 0.2 * kron(Z(0), Z(1)) + 0.3 * kron(X(2), X(3)),
        0.5 * add(Z(0), Z(1), kron(X(2), X(3))) + 0.2 * add(X(2), X(3)),
        add(0.1 * kron(Z(0), Z(1)), 0.2 * kron(X(2), X(3))),
        total_magnetization(4),
        0.1 * kron(Z(0), Z(1)) + 2 * CNOT(0, 1),
    ],
)
def test_check_with_hamevo_exact_fixed_generator(generator: AbstractBlock) -> None:
    paulis = get_pauli_blocks(generator)
    primitives = primitive_blocks(generator)
    is_pauli = len(paulis) == len(primitives)

    n_qubits = generator.n_qubits

    tevo = 2.0
    b1 = HamEvo(generator, parameter=tevo)
    if is_pauli:
        b2 = HamEvo(generator, parameter=tevo).digital_decomposition()
    else:
        with pytest.raises(NotImplementedError):
            _ = HamEvo(generator, parameter=tevo).digital_decomposition()
        return

    c1 = QuantumCircuit(n_qubits, b1)
    c2 = QuantumCircuit(n_qubits, b2)

    model1 = QuantumModel(c1, backend=BackendName.PYQTORCH)
    model2 = QuantumModel(c2, backend=BackendName.PYQTORCH)

    wf1 = model1.run({})
    wf2 = model2.run({})

    assert torch.allclose(wf1, wf2, atol=1.0e-7)


@pytest.mark.parametrize(
    "generator",
    [
        kron(X(0), X(1), X(2)),
        chain(chain(chain(chain(X(0))))),
        kron(kron(X(0), kron(X(1))), kron(X(2))),
        chain(kron(X(0), kron(X(1))), kron(X(1))),
        2 * kron(kron(X(0), kron(X(1))), kron(X(2))),
    ],
)
def test_composite_hamevo_edge_cases(generator: AbstractBlock) -> None:
    n_qubits = generator.n_qubits

    tevo = 0.005
    b1 = HamEvo(generator, parameter=tevo)
    b2 = HamEvo(generator, parameter=tevo).digital_decomposition()

    c1 = QuantumCircuit(n_qubits, b1)
    c2 = QuantumCircuit(n_qubits, b2)

    model1 = QuantumModel(c1, backend=BackendName.PYQTORCH)
    model2 = QuantumModel(c2, backend=BackendName.PYQTORCH)

    wf1 = model1.run({})
    wf2 = model2.run({})

    assert torch.allclose(wf1, wf2, atol=1.0e-2)


def open_chem_obs() -> AbstractBlock:
    """A tiny helper function"""
    directory = os.getcwd()
    with open(os.path.join(directory, "tests/test_files/h4.json"), "r") as js:
        obs = loads(js.read())
    return deserialize(obs)  # type: ignore[return-value]


@pytest.mark.parametrize(
    "generator",
    [
        kron(X(0), X(1), X(2), X(3)) + kron(Z(0), Z(1), Y(2), X(3)),
        ising_hamiltonian(2),
        ising_hamiltonian(4),
        zz_hamiltonian(2),
        zz_hamiltonian(4),
        open_chem_obs(),  # H4
    ],
)
def test_check_with_hamevo_approximate(generator: AbstractBlock) -> None:
    def _run(
        generator: AbstractBlock, tevo: float, approximation: LTSOrder = LTSOrder.BASIC
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b1 = HamEvo(generator, parameter=tevo)
        b2 = HamEvo(generator, parameter=tevo).digital_decomposition(approximation=approximation)

        c1 = QuantumCircuit(generator.n_qubits, b1)
        c2 = QuantumCircuit(generator.n_qubits, b2)

        model1 = QuantumModel(c1, backend=BackendName.PYQTORCH)
        model2 = QuantumModel(c2, backend=BackendName.PYQTORCH)

        wf1 = model1.run({})
        wf2 = model2.run({})

        return wf1, wf2

    # short time evolution still works
    tevo_short = 0.005
    wf1, wf2 = _run(generator, tevo_short)
    assert torch.allclose(wf1, wf2, atol=DIGITAL_DECOMP_ACCEPTANCE_HIGH)

    # short time evolution better approximation
    tevo_short = 0.005
    wf1, wf2 = _run(generator, tevo_short, approximation=LTSOrder.ST4)
    assert torch.allclose(wf1, wf2, atol=DIGITAL_DECOMP_ACCEPTANCE_LOW)


def test_check_with_hamevo_parametric_scaleblocks() -> None:
    theta1 = VariationalParameter("theta1")
    theta2 = VariationalParameter("theta2")

    generator = theta1 * kron(X(0), X(1)) + theta1 * theta2 * kron(Z(2), Z(3))
    n_qubits = generator.n_qubits

    tevo = 2.0
    b1 = HamEvo(generator, parameter=tevo)
    b2 = HamEvo(generator, parameter=tevo).digital_decomposition()

    c1 = QuantumCircuit(n_qubits, b1)
    c2 = QuantumCircuit(n_qubits, b2)

    model1 = QuantumModel(c1, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    model2 = QuantumModel(c2, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

    wf1 = model1.run({})
    wf2 = model2.run({})

    assert torch.allclose(wf1, wf2, atol=ATOL_32)
