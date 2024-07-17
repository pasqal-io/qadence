from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import numpy as np
import torch
from braket.circuits import Circuit as BraketCircuit
from braket.devices import LocalSimulator
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import to_list_of_dicts
from qadence.blocks import AbstractBlock, block_to_tensor
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.mitigations.protocols import apply_mitigation
from qadence.noise import Noise
from qadence.noise.protocols import apply_noise
from qadence.overlap import overlap_exact
from qadence.transpile import transpile
from qadence.types import BackendName, Engine
from qadence.utils import Endianness

from .config import Configuration, default_passes
from .convert_ops import convert_block

logger = getLogger(__name__)


def promote_parameters(parameters: dict[str, Tensor | float]) -> dict[str, float]:
    float_params = {}
    for name, value in parameters.items():
        try:
            v = value if isinstance(value, float) else value.item()
            float_params[name] = v
        except ValueError:
            raise ValueError("Currently batching is not supported with Braket digital")
    return float_params


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    name: BackendName = BackendName.BRAKET
    supports_ad: bool = False
    supports_adjoint: bool = False
    # TODO Use native braket adjoint differentiation.
    support_bp: bool = False
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = field(default_factory=Configuration)
    engine: Engine = Engine.TORCH
    logger.debug("Initialised")

    # braket specifics
    # TODO: include it in the configuration?
    _device: LocalSimulator = field(default_factory=LocalSimulator)

    def __post_init__(self) -> None:
        if self.is_remote:
            raise NotImplementedError("Braket backend does not support cloud execution yet")

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        passes = self.config.transpilation_passes
        if passes is None:
            passes = default_passes

        original_circ = circuit
        if len(passes) > 0:
            circuit = transpile(*passes)(circuit)

        native = BraketCircuit(convert_block(circuit.block))
        return ConvertedCircuit(native=native, abstract=circuit, original=original_circ)

    def observable(self, obs: AbstractBlock, n_qubits: int = None) -> Any:
        if n_qubits is None:
            n_qubits = obs.n_qubits
        native = block_to_tensor(
            block=obs,
            values={},
            qubit_support=tuple([i for i in range(n_qubits)]),
            endianness=Endianness.BIG,
        ).squeeze(0)
        return ConvertedObservable(native=native, abstract=obs, original=obs)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        """
        Execute the circuit and return a wavefunction in form of a statevector.

        Arguments:
            circuit: The circuit that is executed.
            param_values: Parameters of the circuit (after calling the embedding
                function on the user-facing parameters).
            state: Initial state.
            endianness: The endianness of the wave function.
        """

        if state is not None:
            raise NotImplementedError

        if self.is_remote:
            # handle here, or different backends?
            raise NotImplementedError

        # loop over all values in the batch
        results = []
        for vals in to_list_of_dicts(param_values):
            final_circuit = self.assign_parameters(circuit, vals)

            final_circuit.state_vector()  # set simulation type
            task = self._device.run(final_circuit, 0)
            results.append(task.result().values[0])
        states = torch.tensor(np.array(results))

        n_qubits = circuit.abstract.n_qubits
        if endianness != self.native_endianness and n_qubits > 1:
            from qadence.transpile import invert_endianness

            states = invert_endianness(states)
        return states

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
        """Execute the circuit and return samples of the resulting wavefunction."""
        if state is not None:
            raise NotImplementedError("Braket cannot handle a custom initial state.")

        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        if self.is_remote:
            # handle here, or different backends?
            raise NotImplementedError

        # loop over all values in the batch

        samples = []
        for vals in to_list_of_dicts(param_values):
            final_circuit = self.assign_parameters(circuit, vals)
            task = self._device.run(final_circuit, n_shots)
            samples.append(task.result().measurement_counts)
        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            samples = invert_endianness(samples)
        if noise is not None:
            samples = apply_noise(noise=noise, samples=samples)
        if mitigation is not None:
            logger.warning(
                "Mitigation protocol is deprecated. Use qadence-protocols instead.",
            )
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
        # Noise is ignored if measurement protocol is not provided.
        if noise is not None and measurement is None:
            logger.warning(
                f"Errors of type {noise} are not implemented for exact expectation yet. "
                "This is ignored for now."
            )
        # Do not flip endianness here because then we would have to reverse the observable
        wfs = self.run(circuit, param_values, state=state, endianness=Endianness.BIG)

        # TODO: Handle batching
        res = []
        observable = observable if isinstance(observable, list) else [observable]
        for wf in wfs:
            res.append([torch.vdot(wf, obs.native @ wf).real for obs in observable])

        return torch.tensor(res)

    def assign_parameters(
        self, circuit: ConvertedCircuit, param_values: dict[str, Tensor | float]
    ) -> BraketCircuit:
        """Assign numerical values to the circuit parameters."""
        if param_values is None:
            return circuit.native()

        params_copy = param_values.copy()
        pnames = [p.name for p in circuit.native.parameters]

        # account for fixed parameters
        for name in param_values.keys():
            if name not in pnames:
                params_copy.pop(name)

        # make sure that all the parameters are single floats
        # otherwise it won't be accepted by Braket
        native_params = promote_parameters(params_copy)

        # assign the parameters to the circuit
        assigned_circuit = circuit.native(**native_params)

        return assigned_circuit

    @staticmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        return overlap_exact(bras, kets)

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
