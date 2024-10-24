from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import torch
from pyqtorch.differentiation import AdjointExpectation
from torch import Tensor
from torch.autograd import Function

from qadence.backend import Backend as QuantumBackend
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import infer_batchsize, is_pyq_shape, param_dict, pyqify, validate_state
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.utils import uuid_to_eigen
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.ml_tools import promote_to_tensor
from qadence.noise import NoiseHandler
from qadence.types import Endianness


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
            # Check for first element being a list in case of noisy simulations in Pulser.
            if isinstance(expectation_values[0], list):
                exp_vals: list = []
                for expectation_value in expectation_values:
                    res = list(map(lambda x: x.get_final_state().data.toarray(), expectation_value))
                    exp_vals.append(torch.tensor(res))
                expectation_values = exp_vals
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
    noise: NoiseHandler | None = None
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
                backend=self.backend,
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
                self.state,
                self.observable[0].native,  # Currently, adjoint only supports a single observable.
                None,
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
            self.circuit.abstract, [o.abstract for o in self.observable], psr_fn, **psr_args
        )

        # Select the subset of all parameters for which PSR apply
        # which are from the circuit only.
        self.param_values = {k: self.param_values[k] for k in param_to_psr.keys()}

        return PSRExpectation.apply(expectation_fn, param_to_psr.values(), self.param_values.keys(), *self.param_values.values())  # type: ignore # noqa: E501

    # Make PSR construction a static method to avoid unhashability issues.
    @staticmethod
    def construct_rules(
        circuit: QuantumCircuit,
        observable: list[AbstractBlock],
        psr_fn: Callable,
        **psr_args: float | None,
    ) -> dict[str, Callable]:
        """Create a mapping between parameters and PSR functions."""

        uuid_to_eigs = uuid_to_eigen(circuit.block, rescale_eigenvals_timeevo=True)
        # We currently rely on implicit ordering to match the PSR to the parameter,
        # because we want to cache PSRs.

        param_to_psr = OrderedDict()
        for param_id, eigenvalues_shift_factor in uuid_to_eigs.items():
            eigenvalues, shift_factor = eigenvalues_shift_factor
            if eigenvalues is None:
                raise ValueError(
                    f"Eigenvalues are not defined for param_id {param_id}\n"
                    # f"of type {type(block)}.\n"
                    "PSR cannot be defined in that case."
                )
            if shift_factor == 1:
                param_to_psr[param_id] = psr_fn(eigenvalues, **psr_args)
            else:
                psr_args_factor = psr_args.copy()
                if "shift_prefac" in psr_args_factor:
                    if psr_args_factor["shift_prefac"] is not None:
                        psr_args_factor["shift_prefac"] = (
                            shift_factor * psr_args_factor["shift_prefac"]
                        )
                else:
                    psr_args_factor["shift_prefac"] = shift_factor
                param_to_psr[param_id] = psr_fn(eigenvalues, **psr_args_factor)
        for obs in observable:
            for param_id, _ in uuid_to_eigen(obs).items():
                # We need the embedded fixed params of the observable in the param_values dict
                # to be able to call expectation. Since torch backward requires
                # a list of param_ids and values of equal length, we need to pass them to PSR too.
                # Since they are constants their gradients are 0.
                param_to_psr[param_id] = lambda x: torch.tensor([0.0], requires_grad=False)
        return param_to_psr
