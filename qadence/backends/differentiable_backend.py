from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from qadence.backend import Backend, Converted, ConvertedCircuit, ConvertedObservable
from qadence.blocks import (
    AbstractBlock,
)
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.types import Endianness, Engine


@dataclass(frozen=True, eq=True)
class DifferentiableBackend(ABC):
    """The abstract class that defines the interface for differentaible backends."""

    engine: Engine
    backend: Backend

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        """Converts an abstract `QuantumCircuit` to the native backend representation.

        Arguments:
            circuit: A circuit, for example: `QuantumCircuit(2, X(0))`

        Returns:
            A converted circuit `c`. You can access the original, arbstract circuit via `c.abstract`
            and the converted (or backend *native*) circuit via `c.native`.
        """
        raise self.backend.circuit(circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        """Converts an abstract observable (which is just an `AbstractBlock`) to the native backend.

        representation.

        Arguments:
            observable: An observable.
            n_qubits: Number of qubits the observable covers. This is typically `circuit.n_qubits`.

        Returns:
            A converted observable `o`. You can access the original, arbstract observable via
            `o.abstract` and the converted (or backend *native*) observable via `o.native`.
        """
        raise self.backend.observable(observable, n_qubits)

    def convert(
        self, circuit: QuantumCircuit, observable: list[AbstractBlock] | AbstractBlock | None = None
    ) -> Converted:
        """Convert an abstract circuit and an optional observable to their native representation.

        Additionally, this function constructs an embedding function which maps from
        user-facing parameters to device parameters (read more on parameter embedding
        [here][qadence.blocks.embedding.embedding]).
        """
        return self.backend.convert(circuit, observable)

    @abstractmethod
    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1000,
        state: Tensor | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        """Sample bit strings.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            n_shots: Number of shots to sample.
            state: Initial state.
            noise: A noise model to use.
            mitigation: An error mitigation protocol to apply.
            endianness: Endianness of the resulting bit strings.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Any:
        """Run a circuit and return the resulting wave function.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            state: Initial state.
            endianness: Endianness of the resulting samples.

        Returns:
            A list of Counter objects where each key represents a bitstring
            and its value the number of times it has been sampled from the given wave function.
        """
        raise NotImplementedError

    @abstractmethod
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
    ) -> Any:
        """Compute the expectation value of the `circuit` with the given `observable`.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            state: Initial state.
            measurement: Optional measurement protocol. If None, use
                exact expectation value with a statevector simulator.
            noise: A noise model to use.
            endianness: Endianness of the resulting bit strings.
        """
        raise NotImplementedError

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: dict[str, Tensor]) -> Any:
        return self.backend.assign_parameters(circuit, param_values)

    @staticmethod
    @abstractmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        raise NotImplementedError
