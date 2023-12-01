from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp
from horqrux.utils import prepare_state
from jax import Array, custom_vjp

from qadence.backend import Backend as QuantumBackend
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.jax_utils import (
    tensor_to_jnp,
)
from qadence.blocks.utils import uuid_to_eigen
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
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
    noise: Noise | None = None
    mitigation: Mitigations | None = None
    endianness: Endianness = Endianness.BIG
    engine: Engine = Engine.JAX

    def psr(self) -> Any:
        n_qubits = self.circuit.abstract.n_qubits
        observable = self.observable[0]
        if self.state is None:
            self.state = prepare_state(n_qubits, "0" * n_qubits)

        def _expectation_fn(state: Array, values: dict, psr_params: dict) -> Array:
            wf = self.circuit.native.forward(state, values)
            return observable.native.forward(wf, values)

        @custom_vjp
        def _expectation(state: Array, values: dict, psr_params: dict) -> Array:
            return _expectation_fn(state, values, psr_params)

        values = self.param_values
        uuid_to_eigs = {
            k: tensor_to_jnp(v) for k, v in uuid_to_eigen(self.circuit.abstract.block).items()
        }
        psr_params = {k: values[k] for k in uuid_to_eigs.keys()}

        def _expectation_fwd(state: Array, values: dict, psr_params: dict) -> Any:
            return _expectation_fn(state, values, psr_params), (
                state,
                values,
                psr_params,
            )

        def _expectation_bwd(res: Tuple[Array, ParamDictType, dict], v: Array) -> Any:
            state, values, psr_params = res
            grads = {}
            # Hardcoding the single spectral_gap to 2. for jax.lax jitting reasons.
            spectral_gap = jnp.array(2.0, dtype=jnp.float64)
            shift = jnp.pi / 2
            for param_name, _ in psr_params.items():
                shifted_values = values.copy()
                shifted_values[param_name] = shifted_values[param_name] + shift
                f_plus = _expectation(state, shifted_values, psr_params)
                shifted_values = values.copy()
                shifted_values[param_name] = shifted_values[param_name] - shift
                f_min = _expectation(state, shifted_values, psr_params)
                grad = spectral_gap * (f_plus - f_min) / (4.0 * jnp.sin(spectral_gap * shift / 2.0))
                grads[param_name] = (v * grad).squeeze()  # Need dimensionless arrays
            return None, None, grads

        _expectation.defvjp(_expectation_fwd, _expectation_bwd)
        return _expectation(self.state, values, psr_params)
