from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module

from qadence.backend import Backend as QuantumBackend
from qadence.backend import Converted, ConvertedCircuit, ConvertedObservable
from qadence.backends.adjoint import AdjointExpectation
from qadence.backends.utils import infer_batchsize, is_pyq_shape, param_dict, pyqify, validate_state
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import uuid_to_block, uuid_to_eigen
from qadence.circuit import QuantumCircuit
from qadence.extensions import get_gpsr_fns
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.ml_tools import promote_to_tensor
from qadence.noise import Noise
from qadence.types import DiffMode, Endianness

logger = get_logger(__name__)


class PSRExpectation(Function):
    """Overloads the PyTorch AD system to perform parameter shift rule on quantum circuits."""

    @staticmethod
    def forward(
        ctx: Any,
        expectation_fn: Callable[[dict[str, Tensor]], Tensor],
        param_psrs: Sequence[Callable],
        param_keys: Sequence[str],
        *param_values: Tensor,
    ) -> Tensor:
        for param in param_values:
            param = param.detach()

        ctx.expectation_fn = expectation_fn
        ctx.param_psrs = param_psrs
        ctx.param_keys = param_keys
        ctx.save_for_backward(*param_values)

        expectation_values = expectation_fn(param_values=param_dict(param_keys, param_values))  # type: ignore[call-arg] # noqa: E501
        # Stack batches of expectations if so.
        if isinstance(expectation_values, list):
            return torch.stack(expectation_values)
        else:
            return expectation_values

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple:
        params = param_dict(ctx.param_keys, ctx.saved_tensors)

        def expectation_fn(params: dict[str, Tensor]) -> Tensor:
            return PSRExpectation.apply(
                ctx.expectation_fn,
                ctx.param_psrs,
                params.keys(),
                *params.values(),
            )

        def vjp(psr: Callable, name: str) -> Tensor:
            """
            !!! warn.

                Sums over gradients corresponding to different observables.
            """
            return (grad_out * psr(expectation_fn, params, name)).sum(dim=1)

        grads = [
            vjp(psr, name) if needs_grad else None
            for psr, name, needs_grad in zip(
                ctx.param_psrs, ctx.param_keys, ctx.needs_input_grad[3:]
            )
        ]
        return (None, None, None, *grads)


@dataclass
class DifferentiableExpectation:
    """A handler for differentiating expectation estimation using various engines."""

    backend: QuantumBackend
    circuit: ConvertedCircuit
    observable: list[ConvertedObservable] | ConvertedObservable
    param_values: dict[str, Tensor]
    state: Tensor | None = None
    measurement: Measurements | None = None
    noise: Noise | None = None
    mitigation: Mitigations | None = None
    endianness: Endianness = Endianness.BIG

    def ad(self) -> Tensor:
        self.observable = (
            self.observable if isinstance(self.observable, list) else [self.observable]
        )
        if self.measurement:
            expectation_fn = self.measurement.get_measurement_fn()
            expectations = expectation_fn(
                circuit=self.circuit.original,
                observables=[obs.original for obs in self.observable],
                param_values=self.param_values,
                options=self.measurement.options,
                state=self.state,
                noise=self.noise,
                endianness=self.endianness,
            )
        else:
            expectations = self.backend.expectation(
                circuit=self.circuit,
                observable=self.observable,
                param_values=self.param_values,
                state=self.state,
                noise=self.noise,
                mitigation=self.mitigation,
                endianness=self.endianness,
            )
        return promote_to_tensor(
            expectations if isinstance(expectations, Tensor) else torch.tensor(expectations)
        )

    def adjoint(self) -> Tensor:
        self.observable = (
            self.observable if isinstance(self.observable, list) else [self.observable]
        )
        if len(self.observable) > 1:
            raise NotImplementedError("AdjointExpectation currently only supports one observable.")

        n_qubits = self.circuit.abstract.n_qubits
        values_batch_size = infer_batchsize(self.param_values)
        if self.state is None:
            self.state = self.circuit.native.init_state(batch_size=values_batch_size)
        else:
            validate_state(self.state, n_qubits)
            self.state = (
                pyqify(self.state, n_qubits)
                if not is_pyq_shape(self.state, n_qubits)
                else self.state
            )
        batch_size = max(values_batch_size, self.state.size(-1))
        return (
            AdjointExpectation.apply(
                self.circuit.native,
                self.observable[0].native,  # Currently, adjoint only supports a single observable.
                self.state,
                self.param_values.keys(),
                *self.param_values.values(),
            )
            .unsqueeze(1)
            .reshape(batch_size, 1)
        )  # we expect (batch_size, n_observables) shape

    def psr(self, psr_fn: Callable, **psr_args: int | float | None) -> Tensor:
        # wrapper which unpacks the parameters
        # as pytorch grads can only calculated w.r.t tensors
        # so we unpack the params, feed in the names separately
        # as apply doesn't take keyword arguments
        # We also fold in the observable into the backend which makes
        # life easier in the custom autodiff.
        self.observable = (
            self.observable if isinstance(self.observable, list) else [self.observable]
        )

        if self.measurement is not None:
            expectation_fn = partial(
                self.measurement.get_measurement_fn(),
                circuit=self.circuit.original,
                observables=[obs.original for obs in self.observable],
                options=self.measurement.options,
                state=self.state,
                noise=self.noise,
                endianness=self.endianness,
            )
        else:
            expectation_fn = partial(
                self.backend.expectation,
                circuit=self.circuit,
                observable=self.observable,
                state=self.state,
                noise=self.noise,
                mitigation=self.mitigation,
                endianness=self.endianness,
            )
        # PSR only applies to parametric circuits.
        if isinstance(self.observable, ConvertedObservable):
            self.observable = [self.observable]
        param_to_psr = self.construct_rules(
            self.circuit,
            [o for o in self.observable],
            psr_fn,
            bknd_expfn=self.backend.expectation,
            **psr_args,
        )

        # Select the subset of all parameters for which PSR apply
        # which are from the circuit only.
        self.param_values = {k: self.param_values[k] for k in param_to_psr.keys()}

        return PSRExpectation.apply(expectation_fn, param_to_psr.values(), self.param_values.keys(), *self.param_values.values())  # type: ignore # noqa: E501

    # Make PSR construction a static method to avoid unhashability issues.
    @staticmethod
    def construct_rules(
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable],
        psr_fn: Callable,
        bknd_expfn: Callable,
        **psr_args: int | float | None,
    ) -> dict[str, Callable]:
        """Create a mapping between parameters and PSR functions."""

        uuid_to_eigs = uuid_to_eigen(circuit.abstract.block)
        # We currently rely on implicit ordering to match the PSR to the parameter,
        # because we want to cache PSRs.

        param_to_psr = OrderedDict()
        for param_id, eigenvalues in uuid_to_eigs.items():
            if eigenvalues is None:
                raise ValueError(
                    f"Eigenvalues are not defined for param_id {param_id}\n"
                    "PSR cannot be defined in that case."
                )

            param_to_psr[param_id] = psr_fn(eigenvalues, **psr_args)

        def ad_expectation(exp_fn: Callable, param_name: str, param_values: dict) -> Tensor:
            expval = bknd_expfn(circuit, observable, param_values)
            return torch.autograd.grad(
                expval, param_values[param_name], torch.ones_like(expval), create_graph=True
            )[0]

        for obs in observable:
            for param_id, _ in uuid_to_block(obs.abstract).items():
                # Trainable parameters in the observable can only be differentiated using AD.
                param_to_psr[param_id] = lambda exp_fn, param_values, param_id: ad_expectation(
                    exp_fn=exp_fn, param_name=param_id, param_values=param_values
                )
        return param_to_psr


class DifferentiableBackend(Module):
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
        super().__init__()

        self.backend = backend
        self.diff_mode = diff_mode
        self.psr_args = psr_args
        # TODO: Add differentiable overlap calculation
        self._overlap: Callable = None  # type: ignore [assignment]

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict = {},
        state: Tensor | None = None,
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
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
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
        param_values: dict[str, Tensor],
        n_shots: int = 1,
        state: Tensor | None = None,
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
            logger.info("PSR cannot be applied to a parametric observable. Using AD.")
        return self.backend.observable(observable, n_qubits)

    def convert(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock | None = None,
    ) -> Converted:
        if self.diff_mode == DiffMode.ADJOINT and observable is not None:
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

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: dict[str, Tensor]) -> Any:
        return self.backend.assign_parameters(circuit, param_values)
