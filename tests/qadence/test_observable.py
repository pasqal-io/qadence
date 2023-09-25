from __future__ import annotations

import strategies as st  # type: ignore
import torch
from hypothesis import given, settings

from qadence import block_to_tensor, total_magnetization
from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ScaleBlock,
    add,
    kron,
)
from qadence.operations import X, Y, Z
from qadence.parameters import VariationalParameter
from qadence.serialization import deserialize


def test_to_tensor() -> None:
    n_qubits = 2

    theta1 = VariationalParameter("theta1", value=0.25)
    theta2 = VariationalParameter("theta2", value=0.5)
    theta3 = VariationalParameter("theta3", value=0.75)

    # following here the convention defined in
    # the humar_readable_params() function
    values = {
        "theta1": 0.25,
        "theta2": 0.5,
        "theta3": 0.75,
    }

    obs1 = add(
        values["theta1"] * kron(X(0), X(1)),
        values["theta2"] * kron(Y(0), Y(1)),
        values["theta3"] * kron(Z(0), Z(1)),
    )
    mat1 = block_to_tensor(obs1)

    g2 = add(theta1 * kron(X(0), X(1)), theta2 * kron(Y(0), Y(1)), theta3 * kron(Z(0), Z(1)))
    mat2 = block_to_tensor(g2)
    assert torch.allclose(mat1, mat2)


def test_scaled_observable_serialization() -> None:
    theta1 = VariationalParameter("theta1")
    theta2 = VariationalParameter("theta2")
    theta3 = VariationalParameter("theta3")

    # following here the convention defined in
    # the humar_readable_params() function
    values = {"theta1_#[0]": 0.25, "theta2_#[0]": 0.5, "theta3_#[0]": 0.75}

    obs1 = add(
        values["theta1_#[0]"] * kron(X(0), X(1)),
        values["theta2_#[0]"] * kron(Y(0), Y(1)),
        values["theta3_#[0]"] * kron(Z(0), Z(1)),
    )

    obs2 = add(theta1 * kron(X(0), X(1)), theta2 * kron(Y(0), Y(1)), theta3 * kron(Z(0), Z(1)))

    d2 = obs2._to_dict()
    obs2_0 = deserialize(d2)
    assert obs2 == obs2_0

    d1 = obs1._to_dict()
    obs1_0 = deserialize(d1)
    assert obs1 == obs1_0


def test_totalmagn_serialization() -> None:
    obs = total_magnetization(2)
    d2 = obs._to_dict()
    obs2_0 = deserialize(d2)
    assert obs == obs2_0


def test_scaled_totalmagn_serialization() -> None:
    theta1 = VariationalParameter("theta1")
    obs = theta1 * total_magnetization(2)
    d2 = obs._to_dict()
    obs2_0 = deserialize(d2)
    assert obs == obs2_0


@given(st.observables())
@settings(deadline=None)
def test_observable_strategy(block: AbstractBlock) -> None:
    assert isinstance(block, (ScaleBlock))
    for block in block.block.blocks:  # type: ignore[attr-defined]
        assert isinstance(block, (ScaleBlock, AddBlock))
