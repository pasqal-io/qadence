from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp
from jax import Array, custom_vjp, vmap

from qadence.backend import Backend as QuantumBackend
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.jax_utils import (
    tensor_to_jnp,
)
from qadence.blocks.utils import uuid_to_eigen
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.types import Endianness, Engine, ParamDictType


def compute_single_gap(eigen_vals: Array, default_val: float = 2.0) -> Array:
    eigen_vals = eigen_vals.reshape(1, 2)
    gaps = jnp.abs(jnp.tril(eigen_vals.T - eigen_vals))
    return jnp.unique(jnp.where(gaps > 0.0, gaps, default_val), size=1)


@dataclass
class DifferentiableExpectation:
    """A handler for differentiating expectation estimation using various engines."""

    backend: QuantumBackend
    circuit: ConvertedCircuit
    observable: list[ConvertedObservable] | ConvertedObservable
    param_values: ParamDictType
    state: Array | None = None
    measurement: Measurements | None = None
    noise: NoiseHandler | None = None
    mitigation: Mitigations | None = None
    endianness: Endianness = Endianness.BIG
    engine: Engine = Engine.JAX

    def psr(self) -> Any:
        n_obs = len(self.observable)

        def expectation_fn(state: Array, values: ParamDictType, psr_params: ParamDictType) -> Array:
            return self.backend.expectation(
                circuit=self.circuit, observable=self.observable, param_values=values, state=state
            )

        @custom_vjp
        def expectation(state: Array, values: ParamDictType, psr_params: ParamDictType) -> Array:
            return expectation_fn(state, values, psr_params)

        uuid_to_eigs = {
            k: tensor_to_jnp(v[0]) for k, v in uuid_to_eigen(self.circuit.abstract.block).items()
        }
        self.psr_params = {
            k: self.param_values[k] for k in uuid_to_eigs.keys()
        }  # Subset of params on which to perform PSR.

        def expectation_fwd(state: Array, values: ParamDictType, psr_params: ParamDictType) -> Any:
            return expectation_fn(state, values, psr_params), (
                state,
                values,
                psr_params,
            )

        def expectation_bwd(res: Tuple[Array, ParamDictType, ParamDictType], tangent: Array) -> Any:
            state, values, psr_params = res
            grads = {}
            # FIXME Hardcoding the single spectral_gap to 2.
            spectral_gap = 2.0
            shift = jnp.pi / 2

            def shift_circ(param_name: str, values: dict) -> Array:
                shifted_values = values.copy()
                shiftvals = jnp.array(
                    [shifted_values[param_name] + shift, shifted_values[param_name] - shift]
                )

                def _expectation(val: Array) -> Array:
                    shifted_values[param_name] = val
                    return expectation(state, shifted_values, psr_params)

                return vmap(_expectation, in_axes=(0,))(shiftvals)

            for param_name, _ in psr_params.items():
                f_plus, f_min = shift_circ(param_name, values)
                grad = spectral_gap * (f_plus - f_min) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                grads[param_name] = jnp.sum(tangent * grad, axis=1) if n_obs > 1 else tangent * grad
            return None, None, grads

        expectation.defvjp(expectation_fwd, expectation_bwd)
        return expectation(self.state, self.param_values, self.psr_params)
