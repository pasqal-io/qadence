from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from metrics import JS_ACCEPTANCE
from pulser.register.register import Register as PulserRegister
from pulser.sequence.sequence import Sequence
from pulser_simulation.simulation import QutipEmulator

from qadence.backends.pulser.backend import make_sequence
from qadence.backends.pulser.config import Configuration
from qadence.backends.pulser.devices import Device, RealisticDevice
from qadence.backends.pulser.pulses import digital_rot_pulse, entangle_pulse
from qadence.blocks import AbstractBlock
from qadence.blocks.analog import Interaction
from qadence.circuit import QuantumCircuit
from qadence.divergences import js_divergence
from qadence.operations import RX, RY, entangle
from qadence.register import Register as QadenceRegister


@pytest.mark.parametrize(
    "Qadence_op, func",
    [
        (RX(0, 1.5), lambda ch: digital_rot_pulse(1.5, 0, ch)),
        (RY(1, 1.5), lambda ch: digital_rot_pulse(1.5, np.pi / 2, ch)),
    ],
)
def test_single_qubit_block_conversion(Qadence_op: AbstractBlock, func: Callable) -> None:
    spacing = 10
    n_qubits = 2
    reg = QadenceRegister(n_qubits)
    circ = QuantumCircuit(reg, Qadence_op)
    config = Configuration(spacing=spacing, device_type=Device.REALISTIC)

    seq1 = make_sequence(circ, config)
    sim1 = QutipEmulator.from_sequence(seq1)
    res1 = sim1.run()
    sample1 = res1.sample_final_state(500)

    reg = PulserRegister.rectangle(1, n_qubits, spacing=spacing)
    seq2 = Sequence(reg, RealisticDevice)
    seq2.declare_channel("local", "rydberg_local")
    seq2.target(Qadence_op.qubit_support, "local")
    pulse = func(seq2.device.channels["rydberg_local"])
    seq2.add(pulse, "local")
    sim2 = QutipEmulator.from_sequence(seq2)
    res2 = sim2.run()
    sample2 = res2.sample_final_state(500)
    assert js_divergence(sample1, sample2) < JS_ACCEPTANCE


@pytest.mark.parametrize(
    "Qadence_op, func",
    [
        (entangle(500), lambda ch: entangle_pulse(500, ch)),
    ],
)
def test_multiple_qubit_block_conversion(Qadence_op: AbstractBlock, func: Callable) -> None:
    spacing = 10
    reg = QadenceRegister(2)
    circ = QuantumCircuit(reg, Qadence_op)
    config = Configuration(spacing=spacing)

    seq1 = make_sequence(circ, config)
    sim1 = QutipEmulator.from_sequence(seq1)
    res1 = sim1.run()
    sample1 = res1.sample_final_state(500)

    reg = PulserRegister.rectangle(1, 2, spacing=spacing)
    seq2 = Sequence(reg, RealisticDevice)
    seq2.declare_channel("global", "rydberg_global")
    seq2.add(func(seq2.device.channels["rydberg_global"]), "global")
    sim2 = QutipEmulator.from_sequence(seq2)
    res2 = sim2.run()
    sample2 = res2.sample_final_state(500)

    assert js_divergence(sample1, sample2) < JS_ACCEPTANCE


def test_interaction() -> None:
    with pytest.raises(ValueError, match="Pulser does not support other interactions than 'Ising'"):
        reg = QadenceRegister(2)
        circ = QuantumCircuit(reg, entangle(100))
        config = Configuration(spacing=10, interaction=Interaction.XY)
        make_sequence(circ, config)
