from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip
import torch
from pulser import Register as PulserRegister
from pulser import Sequence
from pulser_simulation import SimConfig
from pulser_simulation.simresults import SimulationResults
from pulser_simulation.simulation import QutipEmulator
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import to_list_of_dicts
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.mitigations.protocols import apply_mitigation
from qadence.noise import Noise
from qadence.noise.protocols import apply_noise
from qadence.overlap import overlap_exact
from qadence.register import Register
from qadence.transpile import transpile
from qadence.types import BackendName, DeviceType, Endianness

from .channels import GLOBAL_CHANNEL, LOCAL_CHANNEL
from .cloud import get_client
from .config import Configuration
from .convert_ops import convert_observable
from .devices import IdealDevice, RealisticDevice
from .pulses import add_addressing_pattern, add_pulses

logger = get_logger(__file__)


def _convert_init_state(state: Tensor) -> np.ndarray:
    """Flip and squeeze initial state consistent with Pulser convention."""
    if state.shape[0] > 1:
        raise ValueError("Pulser backend only supports initial states with batch size 1.")
    return np.flip(state.cpu().squeeze().numpy())


def create_register(register: Register) -> PulserRegister:
    """Convert Qadence Register to Pulser Register."""
    coords = np.array(list(register.coords.values()))
    return PulserRegister.from_coordinates(coords)


def make_sequence(circ: QuantumCircuit, config: Configuration) -> Sequence:
    qadence_register = circ.register
    device_specs = qadence_register.device_specs

    if device_specs.type == DeviceType.IDEALIZED:
        device = IdealDevice(
            device_specs.rydberg_level, device_specs.max_detuning, device_specs.max_amp
        )
    elif device_specs.type == DeviceType.REALISTIC:
        device = RealisticDevice(
            device_specs.rydberg_level, device_specs.max_detuning, device_specs.max_amp
        )
    else:
        raise ValueError(
            f"Specified device of type {device_specs.type} is not supported by the pulser backend."
        )

    ########
    # FIXME: Remove the block below in V1.1.0
    if config.spacing is not None:
        logger.warning(
            "Passing register spacing in the backend configuration is deprecated. "
            "Please pass it in the register directly, as detailed in the register tutorial."
        )
        # Rescales the register coordinates, as was done with the previous "spacing" argument.
        qadence_register = qadence_register.rescale_coords(scaling=config.spacing)
    else:
        if qadence_register.min_distance < 4.0:
            # Throws warning for minimum distance below 4 because the typical values used
            # for the standard pulser device parameters is ~7-8, so this likely means the user
            # forgot to set the spacing at register creation.
            logger.warning(
                "Register with distance between atoms smaller than 4µm detected. "
                "Pulser backend no longer has a default spacing of 8µm applied to the register. "
                "Make sure you set the desired spacing as detailed in the register tutorial."
            )
    ########

    pulser_register = create_register(qadence_register)

    sequence = Sequence(pulser_register, device)

    sequence.declare_channel(GLOBAL_CHANNEL, "rydberg_global")
    sequence.declare_channel(LOCAL_CHANNEL, "rydberg_local", initial_target=0)

    add_pulses(sequence, circ.block, config, qadence_register)

    return sequence


# TODO: make it parallelized
def simulate_sequence(
    sequence: Sequence, config: Configuration, state: Tensor, n_shots: int | None = None
) -> SimulationResults | Counter:
    if config.cloud_configuration is not None:
        client = get_client(config.cloud_configuration)

        serialized_sequence = sequence.to_abstract_repr()
        params: list[dict] = [{"runs": n_shots, "variables": {}}]

        batch = client.create_batch(
            serialized_sequence,
            jobs=params,
            emulator=str(config.cloud_configuration.platform),
            wait=True,
        )

        job = list(batch.jobs.values())[0]
        if job.errors is not None:
            logger.error(
                f"The cloud job with ID {job.id} has "
                f"failed for the following reason: {job.errors}"
            )

        return Counter(job.result)

    else:
        simulation = QutipEmulator.from_sequence(
            sequence,
            sampling_rate=config.sampling_rate,
            config=config.sim_config,
            with_modulation=config.with_modulation,
        )
        if state is not None:
            simulation.set_initial_state(qutip.Qobj(state))

        sim_result = simulation.run(nsteps=config.n_steps_solv, method=config.method_solv)
        if n_shots is not None:
            return sim_result.sample_final_state(n_shots)
        else:
            return sim_result


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    """The Pulser backend."""

    name: BackendName = BackendName.PULSER
    supports_ad: bool = False
    supports_adjoint: bool = False
    support_bp: bool = False
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = field(default_factory=Configuration)

    def circuit(self, circ: QuantumCircuit) -> Sequence:
        passes = self.config.transpilation_passes
        original_circ = circ
        if passes is not None and len(passes) > 0:
            circ = transpile(*passes)(circ)

        native = make_sequence(circ, self.config)

        return ConvertedCircuit(native=native, abstract=circ, original=original_circ)

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

    def _run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        vals = to_list_of_dicts(param_values)

        # TODO: relax this constraint
        if self.config.cloud_configuration is not None:
            raise ValueError(
                "Cannot retrieve the wavefunction from cloud simulations. Do not"
                "specify any cloud credentials to use the .run() method"
            )

        state = state if state is None else _convert_init_state(state)
        batched_wf = np.zeros((len(vals), 2**circuit.abstract.n_qubits), dtype=np.complex128)

        for i, param_values_el in enumerate(vals):
            sequence = self.assign_parameters(circuit, param_values_el)
            pattern = circuit.original.register.device_specs.pattern
            if pattern is not None:
                add_addressing_pattern(sequence, pattern)
            sequence.measure()
            sim_result = simulate_sequence(sequence, self.config, state, n_shots=None)
            wf = (
                sim_result.get_final_state(  # type:ignore [union-attr]
                    ignore_global_phase=False, normalize=True
                )
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

    def run_dm(
        self,
        circuit: ConvertedCircuit,
        noise: Noise,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list:
        vals = to_list_of_dicts(param_values)
        noise_probas = noise.options.get("noise_probas", None)
        if noise_probas is None:
            KeyError(f"A range of noise probabilies should be passed. Got {noise_probas}.")

        noisy_batched_dm = []

        for noise_proba in noise_probas:
            batched_dm = []
            sim_config = {"noise": noise.protocol, noise.protocol + "_prob": noise_proba}
            self.config.sim_config = SimConfig(**sim_config)

            for i, param_values_el in enumerate(vals):
                sequence = self.assign_parameters(circuit, param_values_el)
                sim_result = simulate_sequence(sequence, self.config, state)
                batched_dm.append(sim_result)
            noisy_batched_dm.append(batched_dm)

        return noisy_batched_dm

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1,
        state: Tensor | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        vals = to_list_of_dicts(param_values)
        state = state if state is None else _convert_init_state(state)

        samples = []
        for param_values_el in vals:
            sequence = self.assign_parameters(circuit, param_values_el)
            pattern = circuit.original.register.device_specs.pattern
            if pattern is not None:
                add_addressing_pattern(sequence, pattern)
            sequence.measure()
            sample = simulate_sequence(sequence, self.config, state, n_shots=n_shots)
            samples.append(sample)
        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            samples = invert_endianness(samples)
        if noise is not None:
            samples = apply_noise(noise=noise, samples=samples)
        if mitigation is not None:
            assert noise
            samples = apply_mitigation(noise=noise, mitigation=mitigation, samples=samples)
        return samples

    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        observables = observable if isinstance(observable, list) else [observable]
        if mitigation is None:
            state = self.run(circuit, param_values=param_values, state=state, endianness=endianness)
            support = sorted(list(circuit.abstract.register.support))
            res_list = [
                obs.native(state, param_values, qubit_support=support) for obs in observables
            ]
            res = torch.transpose(torch.stack(res_list), 0, 1)
            res = res if len(res.shape) > 0 else res.reshape(1)
            return res.real
        elif mitigation is not None:
            mitigation_fn = mitigation.get_mitigation_fn()
            mitigated_exp_val = mitigation_fn(
                backend_name=self.name,
                circuit=circuit,
                observable=observables,
                state=state,
                measurement=measurement,
                noise=noise,
                mitigation=mitigation,
                endianness=endianness,
            )
            return mitigated_exp_val

    @staticmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        return overlap_exact(bras, kets)

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
