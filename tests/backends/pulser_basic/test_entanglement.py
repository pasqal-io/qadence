from __future__ import annotations

from collections import Counter

import pytest
import torch
from metrics import JS_ACCEPTANCE

from qadence import sample
from qadence.backend import BackendName
from qadence.backends.pulser import Device
from qadence.blocks import AbstractBlock, chain
from qadence.divergences import js_divergence
from qadence.operations import RY, entangle
from qadence.register import Register


@pytest.mark.parametrize(
    "blocks,register,goal",
    [
        # Bell state
        (
            chain(entangle(383, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2)),
            Register(2),
            Counter({"00": 250, "11": 250}),
        ),
    ],
)
def test_entanglement(blocks: AbstractBlock, register: Register, goal: Counter) -> None:
    config = {"device_type": Device.REALISTIC}
    res = sample(register, blocks, backend=BackendName.PULSER, n_shots=500, configuration=config)[0]
    assert js_divergence(res, goal) < JS_ACCEPTANCE
