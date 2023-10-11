from __future__ import annotations

from typing import Tuple

import pytest
import strategies as st
from hypothesis import given, settings
from sympy import acos

from qadence import Parameter, QuantumCircuit
from qadence.blocks import AbstractBlock, chain, kron
from qadence.blocks.utils import assert_same_block, put
from qadence.constructors import hea
from qadence.operations import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    # AnEntanglement,
    # AnFreeEvo,
    # AnRX,
    # AnRY,
    H,
    HamEvo,
    I,
    S,
    SDagger,
    T,
    TDagger,
    X,
    Y,
    Z,
    Zero,
)


@pytest.mark.parametrize(
    "block",
    [
        X(0),
        Y(0),
        Z(0),
        S(0),
        SDagger(0),
        T(0),
        TDagger(0),
        CNOT(0, 1),
        CZ(0, 1),
        SWAP(0, 1),
        H(0),
        I(0),
        Zero(),
    ],
)
def test_all_fixed_primitive_blocks(block: AbstractBlock) -> None:
    # testing all fixed primitive blocks, for which U=U''
    assert_same_block(block, block.dagger().dagger())


@pytest.mark.parametrize(
    "block",
    [
        X(0),
        Y(0),
        Z(0),
        I(0),
        H(0),
        CNOT(0, 1),
        CZ(0, 1),
        SWAP(0, 1),
        Zero(),
    ],
)
def test_self_adjoint_blocks(block: AbstractBlock) -> None:
    # some cases are self-adjoint, which means the property U=U'
    assert_same_block(block, block.dagger())


def test_t_and_s_gates() -> None:
    # testing those cases which are not self-adjoint, and require special backend implementations
    assert_same_block(S(0), SDagger(0).dagger())
    assert_same_block(SDagger(0), S(0).dagger())
    assert_same_block(T(0), TDagger(0).dagger())
    assert_same_block(TDagger(0), T(0).dagger())


def test_scale_dagger() -> None:
    # testing scale blocks with numerical or parametric values
    for scale in [2, 2.1, Parameter("x"), acos(Parameter("x"))]:
        assert_same_block(scale * X(0), (-scale * X(0)).dagger())
        assert_same_block(scale * X(0), (scale * X(0)).dagger().dagger())


@pytest.mark.parametrize(
    "block",
    [
        (1, RX),
        (1, RY),
        (1, RZ),
        (2, CRX),
        (2, CRY),
        (2, CRZ),
        (2, CPHASE),
        # (0, AnEntanglement),
        # (0, AnFreeEvo),
        # (0, AnRX),
        # (0, AnRY),
        (-1, HamEvo),
    ],
)
def test_all_self_adjoint_blocks(block: Tuple[int, AbstractBlock]) -> None:
    n_qubits, block_class = block
    for p_type in [1.42, "x", Parameter("x"), acos(Parameter("x"))]:
        if n_qubits >= 0:
            block = block_class(*tuple(range(n_qubits)), p_type)  # type: ignore[operator]
        else:
            generator = X(0) + 3 * Y(1) * Z(1) + 2 * X(1)
            block = HamEvo(generator, p_type)  # type: ignore[assignment]
        assert_same_block(block, block.dagger().dagger())  # type: ignore[arg-type,attr-defined]
        if not isinstance(p_type, str):
            block_dagger = (
                block_class(*tuple(range(n_qubits)), -p_type)  # type: ignore[operator]
                if n_qubits >= 0
                else HamEvo(generator, -p_type)
            )
            assert_same_block(block, block_dagger.dagger())  # type: ignore[arg-type,attr-defined]
            assert_same_block(block.dagger(), block_dagger)  # type: ignore[arg-type,attr-defined]


@pytest.mark.parametrize(
    "block",
    [
        chain(X(0), Y(0), Z(0), Y(0)),
        kron(X(1), Y(3), Z(4), Y(2)),
        chain(kron(X(0), Y(1)), kron(Z(3), H(1))),
        chain(CNOT(0, 1), CNOT(1, 0)),
        X(0) + Y(1),
        X(0) + 3.0 * Y(1),
        hea(3, 2),
        put(X(0), 1, 3),
        # TODO add QFT here
    ],
)
def test_composite_blocks_no_fails(block: AbstractBlock) -> None:
    assert isinstance(block.dagger(), AbstractBlock)


@given(st.restricted_circuits())
@settings(deadline=None)
def test_circuit_dagger(circuit: QuantumCircuit) -> None:
    circuit == circuit.dagger().dagger()
