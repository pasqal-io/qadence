from __future__ import annotations

from qadence.backend import Backend, ConvertedCircuit, ConvertedObservable
from qadence.engines.differentiable_backend import (
    DifferentiableBackend as DifferentiableBackendInterface,
)
from qadence.engines.jax.differentiable_expectation import DifferentiableExpectation
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.types import ArrayLike, DiffMode, Endianness, Engine, ParamDictType


class DifferentiableBackend(DifferentiableBackendInterface):
    """A class to abstract the operations done by the autodiff engine.

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
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> ArrayLike:
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
