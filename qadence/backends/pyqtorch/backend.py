from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import pyqtorch as pyq
import torch
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import (
    infer_batchsize,
    pyqify,
    to_list_of_dicts,
    unpyqify,
    validate_state,
)
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations.protocols import Mitigations, apply_mitigation
from qadence.noise import NoiseHandler
from qadence.transpile import (
    chain_single_qubit_ops,
    flatten,
    invert_endianness,
    scale_primitive_blocks_only,
    set_noise,
    transpile,
)
from qadence.types import BackendName, Endianness, Engine

from .config import Configuration, default_passes
from .convert_ops import convert_block, convert_readout_noise

logger = getLogger(__name__)


def set_noise_abstract_to_native(circuit: ConvertedCircuit, config: Configuration) -> None:
    """Set noise in native blocks from the abstract ones with noise.

    Args:
        circuit (ConvertedCircuit): Input converted circuit.
    """
    ops = convert_block(circuit.abstract.block, n_qubits=circuit.native.n_qubits, config=config)
    circuit.native = pyq.QuantumCircuit(circuit.native.n_qubits, ops, circuit.native.readout_noise)


def set_readout_noise(circuit: ConvertedCircuit, noise: NoiseHandler) -> None:
    """Set readout noise in place in native.

    Args:
        circuit (ConvertedCircuit):  Input converted circuit.
        noise (NoiseHandler | None): Noise.
    """
    readout = convert_readout_noise(circuit.abstract.n_qubits, noise)
    if readout:
        circuit.native.readout_noise = readout


def set_block_and_readout_noises(
    circuit: ConvertedCircuit, noise: NoiseHandler | None, config: Configuration
) -> None:
    """Add noise on blocks and readout on circuit.

    We first start by adding noise to the abstract blocks. Then we do a conversion to their
    native representation. Finally, we add readout.

    Args:
        circuit (ConvertedCircuit): Input circuit.
        noise (NoiseHandler | None): Noise to add.
    """
    if noise:
        set_noise(circuit, noise)
        set_noise_abstract_to_native(circuit, config)
        set_readout_noise(circuit, noise)


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    """PyQTorch backend."""

    name: BackendName = BackendName.PYQTORCH
    supports_ad: bool = True
    support_bp: bool = True
    supports_adjoint: bool = True
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = field(default_factory=Configuration)
    engine: Engine = Engine.TORCH
    logger.debug("Initialised")

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        """Return the converted circuit.

        Note that to get a representation with noise, noise
        should be passed within the config.

        Args:
            circuit (QuantumCircuit): Original circuit

        Returns:
            ConvertedCircuit: ConvertedCircuit instance for backend.
        """
        passes = self.config.transpilation_passes
        if passes is None:
            passes = default_passes(self.config)

        original_circ = circuit
        if len(passes) > 0:
            circuit = transpile(*passes)(circuit)
        # Setting noise in the circuit.
        if self.config.noise:
            set_noise(circuit, self.config.noise)

        ops = convert_block(circuit.block, n_qubits=circuit.n_qubits, config=self.config)
        readout_noise = (
            convert_readout_noise(circuit.n_qubits, self.config.noise)
            if self.config.noise
            else None
        )
        if self.config.dropout_probability == 0:
            native = pyq.QuantumCircuit(
                circuit.n_qubits,
                ops,
                readout_noise,
            )
        else:
            native = pyq.DropoutQuantumCircuit(
                circuit.n_qubits,
                ops,
                readout_noise,
                dropout_prob=self.config.dropout_probability,
                dropout_mode=self.config.dropout_mode,
            )
        return ConvertedCircuit(native=native, abstract=circuit, original=original_circ)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        # make sure only leaves, i.e. primitive blocks are scaled
        transpilations = [
            lambda block: (
                chain_single_qubit_ops(block)
                if self.config.use_single_qubit_composition
                else flatten(block)
            ),
            scale_primitive_blocks_only,
        ]
        block = transpile(*transpilations)(observable)  # type: ignore[call-overload]
        operations = convert_block(block, n_qubits, self.config)
        native = pyq.Observable(operations=operations)
        return ConvertedObservable(native=native, abstract=block, original=observable)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
        pyqify_state: bool = True,
        unpyqify_state: bool = True,
    ) -> Tensor:
        n_qubits = circuit.abstract.n_qubits
        if state is None:
            # If no state is passed, we infer the batch_size through the length
            # of the individual parameter value tensors.
            state = circuit.native.init_state(batch_size=infer_batchsize(param_values))
        else:
            validate_state(state, n_qubits)
            # pyqtorch expects input shape [2] * n_qubits + [batch_size]
            state = pyqify(state, n_qubits) if pyqify_state else state
        state = circuit.native.run(state=state, values=param_values)
        state = unpyqify(state) if unpyqify_state else state
        state = invert_endianness(state) if endianness != self.native_endianness else state
        return state

    def _batched_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        set_block_and_readout_noises(circuit, noise, self.config)
        state = self.run(
            circuit,
            param_values=param_values,
            state=state,
            endianness=endianness,
            pyqify_state=True,
            # we are calling  the native observable directly, so we want to use pyq shapes
            unpyqify_state=False,
        )
        observable = observable if isinstance(observable, list) else [observable]
        _expectation = torch.hstack(
            [obs.native.expectation(state, param_values).reshape(-1, 1) for obs in observable]
        )
        return _expectation

    def _looped_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        if state is None:
            from qadence.states import zero_state

            state = zero_state(circuit.abstract.n_qubits, batch_size=1).to(
                dtype=circuit.native.dtype
            )
        if state.size(0) != 1:
            raise ValueError(
                "Looping expectation does not make sense with batched initial state. "
                "Define your initial state with `batch_size=1`"
            )

        set_block_and_readout_noises(circuit, noise, self.config)

        list_expvals = []
        observables = observable if isinstance(observable, list) else [observable]
        for vals in to_list_of_dicts(param_values):
            wf = self.run(circuit, vals, state, endianness, pyqify_state=True, unpyqify_state=False)
            exs = torch.cat([obs.native.expectation(wf, vals) for obs in observables], 0)
            list_expvals.append(exs)

        batch_expvals = torch.vstack(list_expvals)
        return batch_expvals if len(batch_expvals.shape) > 0 else batch_expvals.reshape(1)

    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        # Noise is ignored if measurement protocol is not provided.
        if noise is not None and measurement is None:
            logger.warning(
                f"Errors of type {noise} are not implemented for exact expectation yet. "
                "This is ignored for now."
            )
        fn = self._looped_expectation if self.config.loop_expectation else self._batched_expectation
        return fn(
            circuit=circuit,
            observable=observable,
            param_values=param_values,
            state=state,
            measurement=measurement,
            noise=noise,
            endianness=endianness,
        )

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1,
        state: Tensor | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
        pyqify_state: bool = True,
    ) -> list[Counter]:
        if state is None:
            state = circuit.native.init_state(batch_size=infer_batchsize(param_values))
        elif state is not None and pyqify_state:
            n_qubits = circuit.abstract.n_qubits
            state = pyqify(state, n_qubits) if pyqify_state else state
        set_block_and_readout_noises(circuit, noise, self.config)
        samples: list[Counter] = circuit.native.sample(
            state=state, values=param_values, n_shots=n_shots
        )
        samples = invert_endianness(samples) if endianness != Endianness.BIG else samples
        if mitigation is not None:
            logger.warning(
                "Mitigation protocol is deprecated. Use qadence-protocols instead.",
            )
            assert noise
            samples = apply_mitigation(noise=noise, mitigation=mitigation, samples=samples)
        return samples

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: dict[str, Tensor]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        from qadence.overlap import overlap_exact

        return overlap_exact(bras, kets)

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
