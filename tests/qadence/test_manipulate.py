from __future__ import annotations

import pytest
import torch
from openfermion import QubitOperator

from qadence import QuantumModel
from qadence.blocks import AbstractBlock, PutBlock, add, chain, kron
from qadence.blocks.manipulate import from_openfermion, to_openfermion
from qadence.circuit import QuantumCircuit
from qadence.constructors import total_magnetization
from qadence.operations import CNOT, CRX, RX, I, X, Y, Z
from qadence.parameters import FeatureParameter
from qadence.transpile import invert_endianness, scale_primitive_blocks_only, validate


@pytest.mark.parametrize(
    "block_and_op",
    [
        (Y(0), QubitOperator("Y0")),
        (X(1), QubitOperator("X1")),
        (add(X(0), X(1)), QubitOperator("X0") + QubitOperator("X1")),
        (
            add(kron(X(0), X(1)), kron(Y(0), Y(1))) * 0.5,
            (QubitOperator("X0 X1") + QubitOperator("Y0 Y1")) * 0.5,
        ),
        (
            chain(kron(X(0), X(1)), kron(Y(2), Y(3))) * 0.5,
            QubitOperator("X0 X1") * QubitOperator("Y2 Y3") * 0.5,
        ),
        (add(X(0), I(1) * 2), QubitOperator("X0") + QubitOperator("", coefficient=2)),
        (X(0) * 1.5j, 1.5j * QubitOperator("X0")),  # type: ignore [operator]
    ],
)
def test_to_openfermion_qubit_operator(block_and_op: tuple) -> None:
    (b, op) = block_and_op
    assert op == to_openfermion(b)
    assert op == to_openfermion(from_openfermion(op))


def test_validate() -> None:
    x = chain(chain(X(0)), chain(X(1)), CRX(2, 3, "phi"), CNOT(2, 3))
    y = validate(x)

    p0, p1 = y.blocks[0], y.blocks[1]  # type: ignore [attr-defined]
    x0 = p0.blocks[0].blocks[0].blocks[0]
    assert isinstance(p0, PutBlock)
    assert p0.qubit_support == (0,)
    assert isinstance(x0, X)
    assert x0.qubit_support == (0,)

    x1 = p1.blocks[0].blocks[0].blocks[0]  # type: ignore [attr-defined]
    assert isinstance(p1, PutBlock)
    assert p1.qubit_support == (1,)
    assert isinstance(x1, X)
    assert x1.qubit_support == (0,)

    x = chain(kron(CNOT(1, 2), CNOT(3, 4)))
    y = validate(x)
    assert y.blocks[0].blocks[0].blocks[1].qubit_support == (2, 3)  # type: ignore[attr-defined]
    assert y.blocks[0].blocks[0].blocks[1].blocks[0].qubit_support == (0, 1)  # type: ignore[attr-defined] # noqa: E501

    b = kron(CNOT(1, 2), CNOT(0, 3))
    y = validate(b)
    assert y.blocks[0].qubit_support == (1, 2)  # type: ignore[attr-defined]
    assert y.blocks[1].qubit_support == (0, 1, 2, 3)  # type: ignore[attr-defined]
    assert y.blocks[0].blocks[0].qubit_support == (0, 1)  # type: ignore[attr-defined]
    assert y.blocks[1].blocks[0].qubit_support == (0, 3)  # type: ignore[attr-defined]


def test_invert_single_scale() -> None:
    b = Z(0) * 1.0
    assert invert_endianness(b, 2, False).qubit_support == (1,)


def test_invert_add_zs() -> None:
    nqubits = 4
    b = add(Z(i) * c for (i, c) in enumerate([1.0] * nqubits))
    b1 = invert_endianness(b, nqubits, False)
    assert b == invert_endianness(b1, nqubits, False)


def test_invert_observable() -> None:
    nqubits = 4
    x = total_magnetization(nqubits)
    x_prime = invert_endianness(x)
    assert x == invert_endianness(x_prime)


def test_invert_nonsymmentrical_obs() -> None:
    x = X(0) + Y(1) + Z(2) + I(3)
    x_prime = invert_endianness(x)
    assert x == invert_endianness(x_prime)


def test_match_inversions() -> None:
    nqubits = 2
    qc = QuantumCircuit(nqubits, RX(1, FeatureParameter("x")))
    qc_rev = invert_endianness(qc)
    assert qc_rev.block.qubit_support == (0,)

    zz = Z(0)
    iz = Z(1)
    zz_inv = invert_endianness(zz, nqubits, False)
    iz_inv = invert_endianness(iz, nqubits, False)
    assert zz_inv.qubit_support == (1,)
    assert iz_inv.qubit_support == (0,)


@pytest.mark.parametrize(
    "block, truth",
    [
        (2.0 * X(0), 2.0 * X(0)),
        # scale only first block because we are multiplying
        (2.0 * chain(X(0), RX(0, "theta")), chain(2.0 * X(0), RX(0, "theta"))),
        (2.0 * kron(X(0), RX(1, "theta")), kron(2.0 * X(0), RX(1, "theta"))),
        # scale all blocks because we are adding
        (2.0 * add(X(0), RX(0, "theta")), add(2.0 * X(0), 2.0 * RX(0, "theta"))),
        (add(2.0 * chain(X(0))), add(chain(2.0 * X(0)))),
        (
            2.0 * chain(add(X(2), 3.0 * X(3)), RX(0, "theta")),
            chain(add(2.0 * X(2), 2.0 * 3.0 * X(3)), RX(0, "theta")),
        ),
        (
            add(3.0 * chain(0.5 * (I(0) - Z(0)), 0.5 * (I(1) - Z(1)))),
            add(chain(1.5 * I(0) - 1.5 * Z(0), 0.5 * I(1) - 0.5 * Z(1))),
        ),
    ],
)
def test_scale_primitive_blocks_only(block: AbstractBlock, truth: AbstractBlock) -> None:
    transformed = scale_primitive_blocks_only(block)
    assert truth == transformed

    n = max(block.qubit_support) + 1
    vals = {"theta": torch.zeros(2)}
    si = torch.ones(2, 2**n, dtype=torch.cdouble)
    m1 = QuantumModel(QuantumCircuit(n, block))
    s1 = m1.run(vals, state=si)
    m2 = QuantumModel(QuantumCircuit(n, transformed))
    s2 = m2.run(vals, state=si)
    assert torch.allclose(s1, s2)
