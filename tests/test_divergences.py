from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from qadence.divergences import js_divergence


@pytest.mark.parametrize(
    "counter_p, counter_q, exp_js",
    [
        (
            Counter({"00": 10, "01": 50, "10": 70, "11": 30}),
            Counter({"00": 10, "01": 50, "10": 70, "11": 30}),
            0.0,
        ),
        (Counter({"00": 10, "01": 50}), Counter({"10": 70, "11": 30}), np.log(2.0)),
    ],
)
def test_js_divergence_fixture(counter_p: Counter, counter_q: Counter, exp_js: float) -> None:
    assert np.isclose(js_divergence(counter_p, counter_q), exp_js)
