from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch import Tensor

from qadence.backends.utils import count_bitstrings


@pytest.mark.parametrize(
    "sample, counter",
    [
        (
            torch.tensor(
                [[1, 1], [0, 0], [1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [1, 0], [0, 1]]
            ),
            Counter({"11": 4, "01": 2, "10": 3, "00": 1}),
        )
    ],
)
def test_count_bitstring(sample: Tensor, counter: Counter) -> None:
    assert count_bitstrings(sample) == counter
