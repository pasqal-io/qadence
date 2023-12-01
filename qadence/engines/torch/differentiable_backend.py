from __future__ import annotations

from functools import partial

from qadence.backend import Backend as QuantumBackend
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.engines.differentiable_backend import (
    DifferentiableBackend as DifferentiableBackendInterface,
)
from qadence.engines.torch.differentiable_expectation import DifferentiableExpectation
from qadence.extensions import get_gpsr_fns
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
        backend: QuantumBackend,
        diff_mode: DiffMode = DiffMode.AD,
        **psr_args: int | float | None,
    ) -> None:
        super().__init__(backend=backend, engine=Engine.TORCH, diff_mode=diff_mode)
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
        """Compute the expectation value of a given observable.

        Arguments:
            circuit: A backend native quantum circuit to be executed.
            observable: A backend native observable to compute the expectation value from.
            param_values: A dict of values for symbolic substitution.
            state: An initial state.
            measurement: A shot-based measurement protocol.
            endianness: Endianness of the state.

        Returns:
            A tensor of expectation values.
        """
        observable = observable if isinstance(observable, list) else [observable]
        differentiable_expectation = DifferentiableExpectation(
            backend=self.backend,
            circuit=circuit,
            observable=observable,
            param_values=param_values,
            state=state,
            measurement=measurement,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

        if self.diff_mode == DiffMode.AD:
            expectation = differentiable_expectation.ad
        elif self.diff_mode == DiffMode.ADJOINT:
            expectation = differentiable_expectation.adjoint
        else:
            try:
                fns = get_gpsr_fns()
                psr_fn = fns[self.diff_mode]
            except KeyError:
                raise ValueError(f"{self.diff_mode} differentiation mode is not supported")
            expectation = partial(differentiable_expectation.psr, psr_fn=psr_fn, **self.psr_args)
        return expectation()
