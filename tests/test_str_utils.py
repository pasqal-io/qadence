from __future__ import annotations

import pytest

from qadence import Z, hamiltonian_factory, kron
from qadence.blocks import AbstractBlock, MatrixBlock, KronBlock
from qadence.utils import blocktree_to_mathematical_expression
import torch


@pytest.mark.parametrize(
    "block, expected_str",
    [
        (Z(0), "Z(0)"),
        (MatrixBlock(torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble), (0,)), "MatrixBlock(0)"),
        (hamiltonian_factory(2, detuning=Z), "(Z(0) + Z(1))"),
        (hamiltonian_factory(2, detuning=Z) + kron(Z(0), Z(1)), "((Z(0) + Z(1)) + (Z(0) ⊗ Z(1)))"),
    ],
)
def test_blocktree_to_mathematical_expression(block: AbstractBlock, expected_str: str) -> None:
    assert blocktree_to_mathematical_expression(block.__rich_tree__()) == expected_str
