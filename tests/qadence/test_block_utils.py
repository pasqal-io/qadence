from __future__ import annotations

import pytest
from sympy import cos, symbols

from qadence.blocks import AbstractBlock
from qadence.blocks.utils import expression_to_uuids, uuid_to_block, uuid_to_expression
from qadence.operations import RX, X, chain

(alpha, beta) = symbols("alpha beta")
gamma = cos(alpha + beta)

blocks = [
    X(0),
    RX(0, 0.5),
    RX(0, "theta"),
    RX(0, gamma),
    chain(RX(0, "theta"), RX(1, "theta")),
    chain(RX(0, gamma), RX(1, gamma * gamma)),
    2 * chain(RX(0, "theta"), RX(1, "theta")),
]


@pytest.mark.parametrize("block,length", zip(blocks, [0, 1, 1, 1, 2, 2, 3]))
def test_uuid_to_block(block: AbstractBlock, length: int) -> None:
    assert len(uuid_to_block(block)) == length


@pytest.mark.parametrize("block,length", zip(blocks, [0, 1, 1, 1, 2, 2, 3]))
def test_uuid_to_expression(block: AbstractBlock, length: int) -> None:
    assert len(uuid_to_expression(block)) == length


@pytest.mark.parametrize("block,length", zip(blocks, [0, 1, 1, 1, 1, 2, 2]))
def test_expression_to_uuids(block: AbstractBlock, length: int) -> None:
    print(expression_to_uuids(block))
    assert len(expression_to_uuids(block)) == length
