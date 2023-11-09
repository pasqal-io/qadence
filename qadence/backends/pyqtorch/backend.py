from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pyqtorch as pyq
import torch
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import (
    to_list_of_dicts,
)
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.noise import Noise
from qadence.noise.protocols import apply
from qadence.overlap import overlap_exact
from qadence.states import zero_state
from qadence.transpile import (
    chain_single_qubit_ops,
    flatten,
    scale_primitive_blocks_only,
    transpile,
)
from qadence.types import BackendName, Endianness
from qadence.utils import int_to_basis

from .config import Configuration, default_passes
from .convert_ops import (
    convert_block,
    convert_observable,
    infer_batchsize,
    pyqify,
    unpyqify,
    validate_pyq_state,
)

logger = get_logger(__name__)


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    """PyQTorch backend."""

    # set standard interface parameters
    name: BackendName = BackendName.PYQTORCH
    supports_ad: bool = True
    support_bp: bool = True
    supports_adjoint: bool = True
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = Configuration()

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        passes = self.config.transpilation_passes
        if passes is None:
            passes = default_passes(self.config)

        original_circ = circuit
        if len(passes) > 0:
            circuit = transpile(*passes)(circuit)

        ops = convert_block(circuit.block, n_qubits=circuit.n_qubits, config=self.config)
        native = pyq.QuantumCircuit(circuit.n_qubits, ops)
        return ConvertedCircuit(native=native, abstract=circuit, original=original_circ)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        # make sure only leaves, i.e. primitive blocks are scaled
        transpilations = [
            lambda block: chain_single_qubit_ops(block)
            if self.config.use_single_qubit_composition
            else flatten(block),
            scale_primitive_blocks_only,
        ]
        block = transpile(*transpilations)(observable)  # type: ignore[call-overload]

        (native,) = convert_observable(block, n_qubits=n_qubits, config=self.config)
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
            # pyqtorch expects input shape [2] * n_qubits + [batch_size]
            state = pyqify(state, n_qubits) if pyqify_state else validate_pyq_state(state, n_qubits)
        state = circuit.native.run(state, param_values)
        # make sure that the batch dimension is the first one, as standard
        # for PyTorch, and not the last one as done in PyQ
        state = unpyqify(state) if unpyqify_state else state

        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            state = invert_endianness(state)
        return state

    def _batched_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
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
            [obs.native(state, param_values).reshape(-1, 1) for obs in observable]
        )
        return _expectation

    def _looped_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        if state is None:
            from qadence.states import zero_state

            state = zero_state(circuit.abstract.n_qubits, batch_size=1)
        if state.size(0) != 1:
            raise ValueError(
                "Looping expectation does not make sense with batched initial state. "
                "Define your initial state with `batch_size=1`"
            )

        list_expvals = []
        observables = observable if isinstance(observable, list) else [observable]
        for vals in to_list_of_dicts(param_values):
            wf = self.run(circuit, vals, state, endianness, pyqify_state=True, unpyqify_state=False)
            exs = torch.cat([obs.native(wf, vals) for obs in observables], 0)
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
        noise: Noise | None = None,
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
        noise: Noise | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        def _sample(_probs: Tensor, n_shots: int, endianness: Endianness, n_qubits: int) -> Counter:
            return Counter(
                {
                    int_to_basis(k=k, n_qubits=n_qubits, endianness=endianness): count.item()
                    for k, count in enumerate(
                        torch.bincount(
                            torch.multinomial(input=_probs, num_samples=n_shots, replacement=True)
                        )
                    )
                    if count > 0
                }
            )

        wf = self.run(circuit=circuit, param_values=param_values, state=state)
        probs = torch.abs(torch.pow(wf, 2))
        samples = list(
            map(
                lambda _probs: _sample(
                    _probs=_probs,
                    n_shots=n_shots,
                    endianness=endianness,
                    n_qubits=circuit.abstract.n_qubits,
                ),
                probs,
            )
        )
        if noise is not None:
            samples = apply(noise=noise, samples=samples)
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
