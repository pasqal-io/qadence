from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import jax
import jax.numpy as jnp
from horqrux.utils import zero_state
from jax.typing import ArrayLike

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.jax_utils import (
    horqify,
    unhorqify,
    uniform_batchsize,
)
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.transpile import flatten, scale_primitive_blocks_only, transpile
from qadence.types import BackendName, Endianness, Engine, ParamDictType
from qadence.utils import int_to_basis

from .config import Configuration, default_passes
from .convert_ops import HorqruxCircuit, convert_block, convert_observable

logger = getLogger(__name__)


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    # set standard interface parameters
    name: BackendName = BackendName.HORQRUX  # type: ignore[assignment]
    supports_ad: bool = True
    supports_adjoint: bool = False
    support_bp: bool = True
    supports_native_psr: bool = False
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = field(default_factory=Configuration)
    engine: Engine = Engine.JAX
    logger.debug("Initialised")

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        passes = self.config.transpilation_passes
        if passes is None:
            passes = default_passes(self.config)

        original_circ = circuit
        if len(passes) > 0:
            circuit = transpile(*passes)(circuit)
        ops = convert_block(circuit.block, n_qubits=circuit.n_qubits, config=self.config)
        return ConvertedCircuit(
            native=HorqruxCircuit(ops), abstract=circuit, original=original_circ
        )

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        transpilations = [
            flatten,
            scale_primitive_blocks_only,
        ]
        block = transpile(*transpilations)(observable)  # type: ignore[call-overload]
        hq_obs = convert_observable(block, n_qubits=n_qubits, config=self.config)
        return ConvertedObservable(native=hq_obs, abstract=block, original=observable)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        state: ArrayLike | None = None,
        endianness: Endianness = Endianness.BIG,
        horqify_state: bool = True,
        unhorqify_state: bool = True,
    ) -> ArrayLike:
        """Propagate a state through a circuit given and retrieve the wave function.

        Args:
            circuit: A ConvertedCircuit object holding the native horqrux circuit.
            param_values: A dict holding the embedded parameters which the native ciruit expects.
            state: The input state.
            measurement: The protocol.
            horqify_state: Reshape the state into [2,2,...]
            unhorqify_state: Flatten the resulting state.

        Returns:
            A jax.Array of shape [2,2,...] if unhorqify=True else shape = [1, 2**n_qubits]
        """
        n_qubits = circuit.abstract.n_qubits
        if state is None:
            state = zero_state(n_qubits)
        else:
            state = horqify(state) if horqify_state else state
        state = circuit.native.forward(state, param_values)
        if endianness != self.native_endianness:
            state = jnp.reshape(state, (1, 2**n_qubits))  # batch_size is always 1
            ls = list(range(2**n_qubits))
            permute_ind = jnp.array([int(f"{num:0{n_qubits}b}"[::-1], 2) for num in ls])
            state = state[:, permute_ind]
        if unhorqify_state:
            state = unhorqify(state)
        return state

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
        """Compute the expectation value of a circuit given a observable.

        Args:
            circuit: A ConvertedCircuit object holding the native horqrux circuit.
            observable: A ConvertedObservable object holding the native horqrux observable.
            param_values: A dict holding the embedded parameters which the native ciruit expects.
            state: The input state.
            measurement: The protocol.
            noise: The Noise.
            mitigation: Mitigation strategy.
            endianness (Endianness): The target endianness of the resulting samples.

        Returns:
            A jax.Array of shape (batch_size, n_observables)
        """
        observable = observable if isinstance(observable, list) else [observable]
        batch_size = max([arr.size for arr in param_values.values()])
        n_obs = len(observable)

        def _expectation(params: ParamDictType) -> ArrayLike:
            out_state = self.run(
                circuit, params, state, endianness, horqify_state=True, unhorqify_state=False
            )
            return jnp.array([o.native.forward(out_state, params) for o in observable])

        if batch_size > 1:  # We vmap for batch_size > 1
            expvals = jax.vmap(_expectation, in_axes=({k: 0 for k in param_values.keys()},))(
                uniform_batchsize(param_values)
            )
        else:
            expvals = _expectation(param_values)
        if expvals.size > 1:
            expvals = jnp.reshape(expvals, (batch_size, n_obs))
        else:
            expvals = jnp.squeeze(
                expvals, 0
            )  # For the case of batch_size == n_obs == 1, we remove the dims
        return expvals

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        n_shots: int = 1,
        state: ArrayLike | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        """Samples from a batch of discrete probability distributions.

        Args:
            circuit: A ConvertedCircuit object holding the native horqrux circuit.
            param_values: A dict holding the embedded parameters which the native ciruit expects.
            n_shots: The number of samples to generate per distribution.
            state: The input state.
            noise: The Noise.
            mitigation: Mitigation strategy.
            endianness (Endianness): The target endianness of the resulting samples.

        Returns:
            A list of Counter objects where each key represents a bitstring
            and its value the number of times it has been sampled from the given wave function.
        """
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        wf = self.run(
            circuit=circuit,
            param_values=param_values,
            state=state,
            horqify_state=True,
            unhorqify_state=False,
        )
        probs = jnp.abs(jnp.float_power(wf, 2.0)).ravel()
        key = jax.random.PRNGKey(0)
        # JAX handles pseudo random number generation by tracking an explicit state via a random key
        # For more details, see https://jax.readthedocs.io/en/latest/random-numbers.html
        samples = jax.vmap(
            lambda subkey: jax.random.choice(
                key=subkey, a=jnp.arange(0, 2**circuit.abstract.n_qubits), p=probs
            )
        )(jax.random.split(key, n_shots))

        return [
            Counter(
                {
                    int_to_basis(
                        k=k, n_qubits=circuit.abstract.n_qubits, endianness=endianness
                    ): count.item()
                    for k, count in enumerate(jnp.bincount(samples))
                    if count > 0
                }
            )
        ]

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: ParamDictType) -> Any:
        raise NotImplementedError

    @staticmethod
    def _overlap(bras: ArrayLike, kets: ArrayLike) -> ArrayLike:
        # TODO
        raise NotImplementedError

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
