from __future__ import annotations

from collections import Counter

import torch
from metrics import JS_ACCEPTANCE

from qadence import sample
from qadence.backend import BackendName
from qadence.backends.pulser.devices import Device, RealisticDevice
from qadence.blocks import chain
from qadence.divergences import js_divergence
from qadence.operations import RY, entangle
from qadence.register import Register


def test_entanglement() -> None:
    block = chain(entangle(1000, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2))

    max_amp = RealisticDevice.channels["rydberg_global"].max_amp
    weak_coupling_const = 1.2
    spacing = weak_coupling_const * RealisticDevice.rydberg_blockade_radius(max_amp)

    register = Register(2, scale=spacing)

    config = {"device_type": Device.REALISTIC}

    res = sample(register, block, backend=BackendName.PULSER, n_shots=500, configuration=config)[0]

    goal = Counter({"00": 250, "11": 250})

    assert js_divergence(res, goal) < JS_ACCEPTANCE
