from __future__ import annotations

from collections import Counter
import pytest

from qadence.mitigation.readout import corrupt


@pytest.mark.parametrize(
    "bitflip, counters",
    [
        (0.5, [Counter({"00": 27, "01": 23, "10": 24, "11": 26})])
    ]
)
def test_bitstring_readout_corruption(bitflip: float, counters: list) -> None:
    print(corrupt(bitflip, counters))
