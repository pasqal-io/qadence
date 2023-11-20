from __future__ import annotations

from collections import Counter
from functools import partial
from typing import Any, Callable

import torch
from torch import Tensor

from qadence.backend import Backend as QuantumBackend
from qadence.backend import Converted, ConvertedCircuit, ConvertedObservable
from qadence.backends.differentiable_backend import DifferentiableBackend
from qadence.backends.engines.torch.torch_expectation import DifferentiableExpectation
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import uuid_to_block
from qadence.circuit import QuantumCircuit
from qadence.extensions import get_gpsr_fns
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.types import DiffMode, Endianness, ParamDictType, ReturnType


class TorchBackend(DifferentiableBackend):
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
        super().__init__(backend=backend, engine=backend.engine)
        self.diff_mode = diff_mode
        self.psr_args = psr_args
        # TODO: Add differentiable overlap calculation
        self._overlap: Callable = None  # type: ignore [assignment]

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        state: ReturnType | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        """Run on the underlying backend."""
        return self.backend.run(
            circuit=circuit, param_values=param_values, state=state, endianness=endianness
        )

    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: ParamDictType = {},
        state: ReturnType | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> ReturnType:
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

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        n_shots: int = 1,
        state: ReturnType | None = None,
        noise: Noise | None = None,
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
        with torch.no_grad():
            return self.backend.sample(
                circuit=circuit,
                param_values=param_values,
                n_shots=n_shots,
                state=state,
                noise=noise,
                mitigation=mitigation,
                endianness=endianness,
            )

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        parametrized_blocks = list(uuid_to_block(circuit.block).values())
        non_prim_blocks = filter(lambda b: not isinstance(b, PrimitiveBlock), parametrized_blocks)
        if len(list(non_prim_blocks)) > 0:
            raise ValueError(
                "The circuit contains non-primitive blocks that are currently not supported by the "
                "PSR differentiable mode."
            )
        return self.backend.circuit(circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        if observable is not None and observable.is_parametric:
            raise ValueError("PSR cannot be applied to a parametric observable.")
        return self.backend.observable(observable, n_qubits)

    def convert(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
    ) -> Converted:
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
        return self.backend.convert(circuit, observable)

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: ParamDictType) -> Any:
        return self.backend.assign_parameters(circuit, param_values)
