from __future__ import annotations

from qadence.backend import Backend, ConvertedCircuit, ConvertedObservable
from qadence.engines.differentiable_backend import (
    DifferentiableBackend as DifferentiableBackendInterface,
)
from qadence.engines.jax.differentiable_expectation import DifferentiableExpectation
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.types import ArrayLike, DiffMode, Endianness, Engine, ParamDictType


class DifferentiableBackend(DifferentiableBackendInterface):
    """A class which wraps a QuantumBackend with the automatic differentation engine JAX.

    Arguments:
        backend: An instance of the QuantumBackend type perform execution.
        diff_mode: A differentiable mode supported by the differentiation engine.
        **psr_args: Arguments that will be passed on to `DifferentiableExpectation`.
    """

    def __init__(
        self,
        backend: Backend,
        diff_mode: DiffMode = DiffMode.AD,
        **psr_args: int | float | None,
    ) -> None:
        super().__init__(backend=backend, engine=Engine.JAX, diff_mode=diff_mode)
        self.psr_args = psr_args

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
    ) -> ArrayLike:
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
        observable = observable if isinstance(observable, list) else [observable]

        if self.diff_mode == DiffMode.AD:
            expectation = self.backend.expectation(circuit, observable, param_values, state)
        else:
            expectation = DifferentiableExpectation(
                backend=self.backend,
                circuit=circuit,
                observable=observable,
                param_values=param_values,
                state=state,
                measurement=measurement,
                noise=noise,
                mitigation=mitigation,
                endianness=endianness,
            ).psr()
        return expectation
