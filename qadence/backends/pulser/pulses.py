from __future__ import annotations

from functools import partial
from typing import Union

import numpy as np
from pulser.channels.base_channel import Channel
from pulser.parametrized.variable import Variable, VariableItem
from pulser.pulse import Pulse
from pulser.sequence.sequence import Sequence
from pulser.waveforms import CompositeWaveform, ConstantWaveform, RampWaveform

from qadence import Register
from qadence.blocks import AbstractBlock, CompositeBlock
from qadence.blocks.analog import (
    AnalogBlock,
    AnalogComposite,
    ConstantAnalogRotation,
    Interaction,
    WaitBlock,
)
from qadence.operations import RX, RY, AnalogEntanglement, OpName
from qadence.parameters import evaluate

from .channels import GLOBAL_CHANNEL, LOCAL_CHANNEL
from .config import Configuration
from .waveforms import SquareWaveform

TVar = Union[Variable, VariableItem]

supported_gates = [
    OpName.ZERO,
    OpName.RX,
    OpName.RY,
    OpName.ANALOGENTANG,
    OpName.ANALOGRX,
    OpName.ANALOGRY,
    OpName.ANALOGRZ,
    OpName.ANALOGSWAP,
    OpName.WAIT,
]


def add_pulses(
    sequence: Sequence,
    block: AbstractBlock,
    config: Configuration,
    qc_register: Register,
    spacing: float,
) -> None:
    # we need this because of the case with a single type of block in a KronBlock
    # TODO: document properly

    n_qubits = len(sequence.register.qubits)

    # define qubit support
    qubit_support = block.qubit_support
    if not isinstance(qubit_support[0], int):
        qubit_support = tuple(range(n_qubits))

    if isinstance(block, AnalogBlock) and config.interaction != Interaction.NN:
        raise ValueError(f"Pulser does not support other interactions than '{Interaction.NN}'")

    local_channel = sequence.device.channels["rydberg_local"]
    global_channel = sequence.device.channels["rydberg_global"]

    rx = partial(digital_rot_pulse, channel=local_channel, phase=0, config=config)
    ry = partial(digital_rot_pulse, channel=local_channel, phase=np.pi / 2, config=config)

    # TODO: lets move those to `@singledipatch`ed functions
    if isinstance(block, WaitBlock):
        # wait if its a global wait
        if block.qubit_support.is_global:
            (uuid, duration) = block.parameters.uuid_param("duration")
            t = evaluate(duration) if duration.is_number else sequence.declare_variable(uuid)
            pulse = Pulse.ConstantPulse(duration=t, amplitude=0, detuning=0, phase=0)
            sequence.add(pulse, GLOBAL_CHANNEL, "wait-for-all")

        # do nothing if its a non-global wait, because that means we are doing a rotation
        # on other qubits
        else:
            support = set(block.qubit_support)
            if not support.issubset(sequence.register.qubits):
                raise ValueError("Trying to wait on qubits outside of support.")

    elif isinstance(block, ConstantAnalogRotation):
        ps = block.parameters
        (a_uuid, alpha) = ps.uuid_param("alpha")
        (w_uuid, omega) = ps.uuid_param("omega")
        (p_uuid, phase) = ps.uuid_param("phase")
        (d_uuid, detuning) = ps.uuid_param("delta")

        a = evaluate(alpha) if alpha.is_number else sequence.declare_variable(a_uuid)
        w = evaluate(omega) if omega.is_number else sequence.declare_variable(w_uuid)
        p = evaluate(phase) if phase.is_number else sequence.declare_variable(p_uuid)
        d = evaluate(detuning) if detuning.is_number else sequence.declare_variable(d_uuid)

        # calculate generator eigenvalues
        block.eigenvalues_generator = block.compute_eigenvalues_generator(
            qc_register, block, spacing
        )

        if block.qubit_support.is_global:
            pulse = analog_rot_pulse(a, w, p, d, global_channel, config)
            sequence.add(pulse, GLOBAL_CHANNEL, protocol="wait-for-all")
        else:
            pulse = analog_rot_pulse(a, w, p, d, local_channel, config)
            sequence.target(qubit_support, LOCAL_CHANNEL)
            sequence.add(pulse, LOCAL_CHANNEL, protocol="wait-for-all")

    elif isinstance(block, AnalogEntanglement):
        (uuid, duration) = block.parameters.uuid_param("duration")
        t = evaluate(duration) if duration.is_number else sequence.declare_variable(uuid)
        sequence.add(
            entangle_pulse(t, global_channel, config), GLOBAL_CHANNEL, protocol="wait-for-all"
        )

    elif isinstance(block, (RX, RY)):
        (uuid, p) = block.parameters.uuid_param("parameter")
        angle = evaluate(p) if p.is_number else sequence.declare_variable(uuid)
        pulse = rx(angle) if isinstance(block, RX) else ry(angle)
        sequence.target(qubit_support, LOCAL_CHANNEL)
        sequence.add(pulse, LOCAL_CHANNEL, protocol="wait-for-all")

    elif isinstance(block, CompositeBlock) or isinstance(block, AnalogComposite):
        for block in block.blocks:
            add_pulses(sequence, block, config, qc_register, spacing)

    else:
        msg = f"The pulser backend currently does not support blocks of type: {type(block)}"
        raise NotImplementedError(msg)


def analog_rot_pulse(
    alpha: TVar | float,
    omega: TVar | float,
    phase: TVar | float,
    detuning: TVar | float,
    channel: Channel,
    config: Configuration | None = None,
) -> Pulse:
    # omega in rad/us; detuning in rad/us
    if config is not None:
        if channel.addressing == "Global":
            max_amp = config.amplitude_global if config.amplitude_global is not None else omega
        elif channel.addressing == "Local":
            max_amp = config.amplitude_local if config.amplitude_local is not None else omega
        max_det = config.detuning if config.detuning is not None else detuning
    else:
        max_amp = omega
        max_det = detuning

    # get pulse duration in ns
    duration = 1000 * abs(alpha) / np.sqrt(omega**2 + detuning**2)

    # create amplitude waveform
    amp_wf = SquareWaveform.from_duration(
        duration=duration,  # type: ignore
        max_amp=max_amp,  # type: ignore[arg-type]
        duration_steps=channel.clock_period,  # type: ignore[attr-defined]
        min_duration=channel.min_duration,
    )

    # create detuning waveform
    det_wf = SquareWaveform.from_duration(
        duration=duration,  # type: ignore
        max_amp=max_det,  # type: ignore[arg-type]
        duration_steps=channel.clock_period,  # type: ignore[attr-defined]
        min_duration=channel.min_duration,
    )

    return Pulse(amplitude=amp_wf, detuning=det_wf, phase=abs(phase))


def entangle_pulse(
    duration: TVar | float, channel: Channel, config: Configuration | None = None
) -> Pulse:
    if config is None:
        max_amp = channel.max_amp
    else:
        max_amp = (
            config.amplitude_global if config.amplitude_global is not None else channel.max_amp
        )

    clock = channel.clock_period
    delay_wf = ConstantWaveform(clock * np.ceil(duration / clock), 0)  # type: ignore
    half_pi_wf = SquareWaveform.from_area(
        area=np.pi / 2,
        max_amp=max_amp,  # type: ignore[arg-type]
        duration_steps=clock,  # type: ignore[attr-defined]
        min_duration=channel.min_duration,
    )

    detuning_wf = RampWaveform(duration=half_pi_wf.duration, start=0, stop=np.pi)
    amplitude = CompositeWaveform(half_pi_wf, delay_wf)
    detuning = CompositeWaveform(detuning_wf, delay_wf)
    return Pulse(amplitude=amplitude, detuning=detuning, phase=np.pi / 2)


def digital_rot_pulse(
    angle: TVar | float, phase: float, channel: Channel, config: Configuration | None = None
) -> Pulse:
    if config is None:
        max_amp = channel.max_amp
    else:
        max_amp = config.amplitude_local if config.amplitude_local is not None else channel.max_amp

    # TODO: Implement reverse rotation for angles bigger than π
    amplitude_wf = SquareWaveform.from_area(
        area=abs(angle),  # type: ignore
        max_amp=max_amp,  # type: ignore[arg-type]
        duration_steps=channel.clock_period,  # type: ignore[attr-defined]
        min_duration=channel.min_duration,
    )

    return Pulse.ConstantDetuning(amplitude=amplitude_wf, detuning=0, phase=phase)
