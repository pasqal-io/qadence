from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip
import torch
from pulser import Register as PulserRegister
from pulser import Sequence
from pulser.pulse import Pulse
from pulser_simulation.simresults import SimulationResults
from pulser_simulation.simulation import QutipEmulator
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import BackendName, ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import to_list_of_dicts
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.overlap import overlap_exact
from qadence.register import Register
from qadence.utils import Endianness

from .channels import GLOBAL_CHANNEL, LOCAL_CHANNEL
from .config import Configuration
from .convert_ops import convert_observable
from .devices import Device, IdealDevice, RealisticDevice
from .pulses import add_pulses

WEAK_COUPLING_CONST = 1.2

DEFAULT_SPACING = 8.0  # µm (standard value)


def create_register(register: Register, spacing: float = DEFAULT_SPACING) -> PulserRegister:
    """Create Pulser register instance.

    Args:
        register (Register): graph representing a register with accompanying coordinate data
        spacing (float): distance between qubits in micrometers

    Returns:
        Register: Pulser register
    """

    # create register from coordinates
    coords = np.array(list(register.coords.values()))
    return PulserRegister.from_coordinates(coords * spacing)


def make_sequence(circ: QuantumCircuit, config: Configuration) -> Sequence:
    if config.device_type == Device.IDEALIZED:
        device = IdealDevice
    elif config.device_type == Device.REALISTIC:
        device = RealisticDevice
    else:
        raise ValueError("Specified device is not supported.")

    max_amp = device.channels["rydberg_global"].max_amp
    min_duration = device.channels["rydberg_global"].min_duration

    if config.spacing is not None:
        spacing = config.spacing
    elif max_amp is not None:
        # Ideal spacing for entanglement gate
        spacing = WEAK_COUPLING_CONST * device.rydberg_blockade_radius(max_amp)  # type: ignore
    else:
        spacing = DEFAULT_SPACING

    pulser_register = create_register(circ.register, spacing)

    sequence = Sequence(pulser_register, device)
    sequence.declare_channel(GLOBAL_CHANNEL, "rydberg_global")
    sequence.declare_channel(LOCAL_CHANNEL, "rydberg_local", initial_target=0)

    # add a minimum duration pulse omega=0 pulse at the beginning for simulation convergence reasons
    # since Pulser's QutipEmulator doesn't allow simulation of sequences with total duration < 4ns
    zero_pulse = Pulse.ConstantPulse(
        duration=max(sequence.device.channels["rydberg_global"].min_duration, 4),
        amplitude=0.0,
        detuning=0.0,
        phase=0.0,
    )
    sequence.add(zero_pulse, GLOBAL_CHANNEL, "wait-for-all")

    add_pulses(sequence, circ.block, config, circ.register, spacing)
    sequence.measure()

    return sequence


# TODO: make it parallelized
# TODO: add execution on the cloud platform
def simulate_sequence(
    sequence: Sequence, config: Configuration, state: Tensor
) -> SimulationResults:
    simulation = QutipEmulator.from_sequence(
        sequence,
        sampling_rate=config.sampling_rate,
        config=config.sim_config,
        with_modulation=config.with_modulation,
    )
    if state is not None:
        simulation.set_initial_state(qutip.Qobj(state.cpu().numpy()))

    return simulation.run(nsteps=config.n_steps_solv, method=config.method_solv)


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    """The Pulser backend"""

    name: BackendName = BackendName.PULSER
    supports_ad: bool = False
    support_bp: bool = False
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = Configuration()

    def circuit(self, circ: QuantumCircuit) -> Sequence:
        native = make_sequence(circ, self.config)

        return ConvertedCircuit(native=native, abstract=circ, original=circ)

    def observable(self, observable: AbstractBlock, n_qubits: int = None) -> Tensor:
        from qadence.transpile import flatten, scale_primitive_blocks_only, transpile

        # make sure only leaves, i.e. primitive blocks are scaled
        block = transpile(flatten, scale_primitive_blocks_only)(observable)

        (native,) = convert_observable(block, n_qubits=n_qubits, config=self.config)
        return ConvertedObservable(native=native, abstract=block, original=observable)

    def assign_parameters(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor],
    ) -> Any:
        if param_values == {} and circuit.native.is_parametrized():
            missing = list(circuit.native.declared_variables.keys())
            raise ValueError(f"Please, provide values for the following parameters: {missing}")

        if param_values == {}:
            return circuit.native

        numpy_param_values = {
            k: v.detach().cpu().numpy()
            for (k, v) in param_values.items()
            if k in circuit.native.declared_variables
        }

        return circuit.native.build(**numpy_param_values)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        vals = to_list_of_dicts(param_values)

        batched_wf = np.zeros((len(vals), 2**circuit.abstract.n_qubits), dtype=np.complex128)

        for i, param_values_el in enumerate(vals):
            sequence = self.assign_parameters(circuit, param_values_el)
            sim_result = simulate_sequence(sequence, self.config, state)
            wf = (
                sim_result.get_final_state(ignore_global_phase=False, normalize=True)
                .full()
                .flatten()
            )

            # We flip the wavefunction coming out of pulser,
            # essentially changing logic 0 with logic 1 in the basis states.
            batched_wf[i] = np.flip(wf)

        batched_wf_torch = torch.from_numpy(batched_wf)

        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            batched_wf_torch = invert_endianness(batched_wf_torch)

        return batched_wf_torch

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1,
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        vals = to_list_of_dicts(param_values)

        samples = []
        for param_values_el in vals:
            sequence = self.assign_parameters(circuit, param_values_el)
            sim_result = simulate_sequence(sequence, self.config, state)
            sample = sim_result.sample_final_state(n_shots)
            samples.append(sample)
        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            samples = invert_endianness(samples)
        return samples

    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        protocol: Measurements | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        state = self.run(circuit, param_values=param_values, state=state, endianness=endianness)

        observables = observable if isinstance(observable, list) else [observable]
        support = sorted(list(circuit.abstract.register.support))
        res_list = [obs.native(state, param_values, qubit_support=support) for obs in observables]

        res = torch.transpose(torch.stack(res_list), 0, 1)
        res = res if len(res.shape) > 0 else res.reshape(1)
        return res.real

    @staticmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        return overlap_exact(bras, kets)

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
