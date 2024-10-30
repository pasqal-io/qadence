from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from qadence.backend import Backend, Converted, ConvertedCircuit, ConvertedObservable
from qadence.blocks import AbstractBlock, PrimitiveBlock
from qadence.blocks.utils import uuid_to_block
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.types import ArrayLike, DiffMode, Endianness, Engine, ParamDictType


@dataclass(frozen=True, eq=True)
class DifferentiableBackend(ABC):
    """The abstract class which wraps any (non)-natively differentiable QuantumBackend.

       in an automatic differentiation engine.

    Arguments:
        backend: An instance of the QuantumBackend type perform execution.
        engine: Which automatic differentiation engine the QuantumBackend runs on.
        diff_mode: A differentiable mode supported by the differentiation engine.
    """

    backend: Backend
    engine: Engine
    diff_mode: DiffMode

    # TODO: Add differentiable overlap calculation
    _overlap: Callable = None  # type: ignore [assignment]

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        n_shots: int = 100,
        state: ArrayLike | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        """Sample bitstring from the registered circuit.

        Arguments:
            circuit: A backend native quantum circuit to be executed.
            param_values: The values of the parameters after embedding
            n_shots: The number of shots. Defaults to 1.
            state: Initial state.
            noise: A noise model to use.
            mitigation: A mitigation protocol to apply to noisy samples.
            endianness: Endianness of the resulting bitstrings.

        Returns:
            An iterable with all the sampled bitstrings
        """

        return self.backend.sample(
            circuit=circuit,
            param_values=param_values,
            n_shots=n_shots,
            state=state,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        state: ArrayLike | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> ArrayLike:
        """Run on the underlying backend."""
        return self.backend.run(
            circuit=circuit, param_values=param_values, state=state, endianness=endianness
        )

    @abstractmethod
    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: ParamDictType = {},
        state: ArrayLike | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Any:
        """Compute the expectation value of the `circuit` with the given `observable`.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            observable: A converted observable as returned by `backend.observable`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            state: Initial state.
            measurement: Optional measurement protocol. If None, use
                exact expectation value with a statevector simulator.
            noise: A noise model to use.
            mitigation: The error mitigation to use.
            endianness: Endianness of the resulting bit strings.
        """
        raise NotImplementedError(
            "A DifferentiableBackend needs to override the expectation method."
        )

    def default_configuration(self) -> Any:
        return self.backend.default_configuration()

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        if self.diff_mode == DiffMode.GPSR:
            parametrized_blocks = list(uuid_to_block(circuit.block).values())
            non_prim_blocks = filter(
                lambda b: not isinstance(b, PrimitiveBlock), parametrized_blocks
            )
            if len(list(non_prim_blocks)) > 0:
                raise ValueError(
                    "The circuit contains non-primitive blocks that are currently\
                    not supported by the PSR differentiable mode."
                )
        return self.backend.circuit(circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        if self.diff_mode != DiffMode.AD and observable is not None:
            msg = (
                f"Differentiation mode '{self.diff_mode}' does not support parametric observables."
            )
            if isinstance(observable, list):
                for obs in observable:
                    if obs.is_parametric:
                        raise ValueError(msg)
            else:
                if observable.is_parametric:
                    raise ValueError(msg)
        return self.backend.observable(observable, n_qubits)

    def convert(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
    ) -> Converted:
        return self.backend.convert(circuit, observable)

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: ParamDictType) -> Any:
        return self.backend.assign_parameters(circuit, param_values)
