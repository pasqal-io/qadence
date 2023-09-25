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
from qadence.operations import RY, AnalogRot, entangle, wait
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
        # Four qubits GHZ state
        (
            chain(
                AnalogRot(duration=100, omega=5 * torch.pi, delta=0, phase=0),
                wait(2300),
                AnalogRot(duration=300, omega=5 * torch.pi, delta=0, phase=0),
            ),
            Register.square(qubits_side=2),
            Counter(
                {
                    "1111": 145,
                    "1110": 15,
                    "1101": 15,
                    "1100": 15,
                    "1011": 15,
                    "1010": 15,
                    "1001": 15,
                    "1000": 15,
                    "0111": 15,
                    "0110": 15,
                    "0101": 15,
                    "0100": 15,
                    "0011": 15,
                    "0010": 15,
                    "0001": 15,
                    "0000": 145,
                }
            ),
        ),
    ],
)
def test_entanglement(blocks: AbstractBlock, register: Register, goal: Counter) -> None:
    config = {"device_type": Device.REALISTIC}
    res = sample(register, blocks, backend=BackendName.PULSER, n_shots=500, configuration=config)[0]
    assert js_divergence(res, goal) < JS_ACCEPTANCE
