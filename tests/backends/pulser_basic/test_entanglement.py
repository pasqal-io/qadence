from __future__ import annotations

from collections import Counter

import pytest
import torch
from metrics import JS_ACCEPTANCE

from qadence import sample
from qadence.backend import BackendName
from qadence.backends.pulser.devices import Device
from qadence.blocks import chain
from qadence.divergences import js_divergence
from qadence.operations import RY, entangle
from qadence.register import Register

DEFAULT_SPACING = 8.0


@pytest.mark.parametrize("device_type", [Device.IDEALIZED, Device.REALISTIC])
def test_entanglement(device_type: Device) -> None:
    block = chain(entangle(1000, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2))

    register = Register.line(2, scale=DEFAULT_SPACING)

    config = {"device_type": device_type}

    res = sample(register, block, backend=BackendName.PULSER, n_shots=500, configuration=config)[0]

    goal = Counter({"00": 250, "11": 250})

    assert js_divergence(res, goal) < JS_ACCEPTANCE
