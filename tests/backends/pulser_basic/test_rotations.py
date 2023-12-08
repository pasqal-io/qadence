from __future__ import annotations

from collections import Counter

import pytest
import torch
from metrics import JS_ACCEPTANCE

from qadence.blocks import AbstractBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.divergences import js_divergence
from qadence.models import QuantumModel
from qadence.operations import RX, RY, AnalogRX, AnalogRY
from qadence.register import Register
from qadence.types import BackendName, DiffMode


@pytest.mark.parametrize(
    "block,goal",
    [
        (RY(0, -torch.pi / 2), Counter({"00": 260, "10": 240})),
        (RY(1, -torch.pi / 2), Counter({"00": 260, "01": 240})),
        (RX(0, -torch.pi / 2), Counter({"00": 260, "10": 240})),
        (RX(1, -torch.pi / 2), Counter({"00": 260, "01": 240})),
    ],
)
def test_single_rotation(block: AbstractBlock, goal: Counter) -> None:
    register = Register.from_coordinates([(-0.5, 0), (0.5, 0)], lattice="line")
    circuit = QuantumCircuit(register, block)
    model_pulser = QuantumModel(
        circuit=circuit, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    sample_pulser = model_pulser.sample(n_shots=500)[0]

    assert js_divergence(sample_pulser, goal) < JS_ACCEPTANCE


@pytest.mark.parametrize(
    "single_rotation,global_rotation",
    [
        (chain(RY(0, -torch.pi / 2), RY(1, -torch.pi / 2)), AnalogRY(-torch.pi / 2)),
        (chain(RX(0, -torch.pi / 2), RX(1, -torch.pi / 2)), AnalogRX(-torch.pi / 2)),
    ],
)
def test_single_rotation_multiple_qubits(
    single_rotation: AbstractBlock, global_rotation: AbstractBlock
) -> None:
    register = Register.from_coordinates([(-0.5, 0), (0.5, 0)], lattice="line", spacing=8.0)

    circuit1 = QuantumCircuit(register, single_rotation)
    model_pulser1 = QuantumModel(
        circuit=circuit1, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    sample1 = model_pulser1.sample(n_shots=500)[0]

    circuit2 = QuantumCircuit(register, global_rotation)
    model_pulser2 = QuantumModel(
        circuit=circuit2, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    sample2 = model_pulser2.sample(n_shots=500)[0]

    assert js_divergence(sample1, sample2) < JS_ACCEPTANCE
