from __future__ import annotations

import pytest

from qadence import Z, hamiltonian_factory, kron, Parameter
from qadence.blocks import AbstractBlock, MatrixBlock
from qadence.utils import block_to_mathematical_expression
import torch


@pytest.mark.parametrize(
    "block, expected_str",
    [
        (Z(0), "Z(0)"),
        (MatrixBlock(torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble), (0,)), "MatrixBlock(0)"),
        (hamiltonian_factory(2, detuning=Z), "(Z(0) + Z(1))"),
        (
            2.0 * hamiltonian_factory(2, detuning=Z) + kron(Z(0), Z(1)),
            "(2.000 * (Z(0) + Z(1)) + (Z(0) ⊗ Z(1)))",
        ),
        (Parameter("param_name") * Z(0), "param_name * Z(0)"),
        (Z(0) * Parameter("param_name"), "param_name * Z(0)"),
    ],
)
def test_block_to_mathematical_expression(block: AbstractBlock, expected_str: str) -> None:
    assert block_to_mathematical_expression(block) == expected_str
