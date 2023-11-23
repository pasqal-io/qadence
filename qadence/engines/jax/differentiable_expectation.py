from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import jax.numpy as jnp
from horqrux.types import Gate
from horqrux.utils import prepare_state
from jax import Array, custom_vjp

from qadence.backend import Backend as QuantumBackend
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import (
    tensor_to_jnp,
)
from qadence.blocks.utils import uuid_to_eigen
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.types import Endianness, Engine, ParamDictType

# Type aliases for target and control indices.
TargetIdx = Tuple[Tuple[int, ...], ...]
ControlIdx = Tuple[Union[None, Tuple[int, ...]], ...]

# State is just an array but this clarifies type annotation
State = Array
Measurement = Array


def is_leaf(subtree: Any) -> bool:
    if isinstance(subtree, Gate):
        return True
    else:
        return False


def compute_gap(eigen_vals: Array) -> Array:
    gaps = jnp.unique(jnp.abs(jnp.tril(eigen_vals - eigen_vals.reshape(-1, 1))))
    # We have to filter out zeros
    return gaps[gaps > 0]


@dataclass
class JaxDifferentiableExpectation:
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
        # assert not isinstance(self.observable, list), 'Lists of observables not supported.'
        assert self.measurement is None, "Measurements are not yet supported by engine JAX."
        # assert is == 1, "Only single observable are supported."
        observable = self.observable[0]
        # batch_size = infer_batchsize(self.param_values)

        if self.state is None:
            self.state = prepare_state(
                self.circuit.abstract.n_qubits, "0" * self.circuit.abstract.n_qubits
            )

        def _expectation_fn(state: Array, values: dict, uuid_to_eigen: dict) -> Array:
            wf = self.circuit.native.forward(state, values)
            return observable.native.forward(wf, values)

        @custom_vjp
        def _expectation(state: Array, values: dict, uuid_to_eigen: dict) -> Array:
            return _expectation_fn(state, values, uuid_to_eigen)

        values = self.param_values
        uuid_to_eigs = {
            k: tensor_to_jnp(v) for k, v in uuid_to_eigen(self.circuit.abstract.block).items()
        }
        values = {k: values[k] for k in uuid_to_eigs.keys()}

        def _expectation_fwd(state: Array, values: dict, uuid_to_eigen: dict) -> Any:
            return _expectation_fn(state, values, uuid_to_eigen), (state, values, uuid_to_eigen)

        shift = jnp.array(jnp.pi / 2, dtype=jnp.float64).item()

        def _expectation_bwd(res: Any, v: Array) -> Any:
            state, values, uuid_to_eigen = res
            grads = {}
            for param_name, eigenvals in uuid_to_eigen.items():
                # + pi/2 shift
                spectral_gap = compute_gap(eigenvals)
                shifted_values = values.copy()
                shifted_values[param_name] = shifted_values[param_name] + shift
                f_plus = _expectation_fn(state, shifted_values, uuid_to_eigen)

                # - pi/2 shift
                shifted_values = values.copy()
                shifted_values[param_name] = shifted_values[param_name] - shift
                f_min = _expectation_fn(state, shifted_values, uuid_to_eigen)

                grad = spectral_gap * (f_plus - f_min) / (4 * jnp.sin(spectral_gap * shift / 2))
                grads[param_name] = (v * grad).squeeze()  # Need dimensionless arrays
            return None, grads, None

        _expectation.defvjp(_expectation_fwd, _expectation_bwd)
        return _expectation(self.state, values, uuid_to_eigs)
