from __future__ import annotations

from collections import Counter

import pytest
from metrics import JS_ACCEPTANCE

from qadence.analog import IdealDevice, RealisticDevice, RydbergDevice
from qadence.backend import BackendName
from qadence.blocks import chain
from qadence.divergences import js_divergence
from qadence.execution import sample
from qadence.operations import RY, entangle
from qadence.register import Register
from qadence.types import PI

DEFAULT_SPACING = 8.0


@pytest.mark.parametrize("device", [IdealDevice(), RealisticDevice()])
def test_entanglement(device: RydbergDevice) -> None:
    block = chain(entangle(1000, qubit_support=(0, 1)), RY(0, 3 * PI / 2))

    register = Register.line(2, spacing=DEFAULT_SPACING, device_specs=device)

    res = sample(register, block, backend=BackendName.PULSER, n_shots=500)[0]

    goal = Counter({"00": 250, "11": 250})

    assert js_divergence(res, goal) < JS_ACCEPTANCE
