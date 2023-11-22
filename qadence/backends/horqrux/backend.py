from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from horqrux.utils import prepare_state
from jax.typing import ArrayLike

from qadence.backend import Backend as BackendInterface
from qadence.backend import ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import (
    infer_batchsize,
    jarr_to_tensor,
    pyqify,
    tensor_to_jnp,
    values_to_jax,
)
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import Noise
from qadence.transpile import invert_endianness
from qadence.types import BackendName, Endianness, Engine, ParamDictType, ReturnType
from qadence.utils import int_to_basis

from .config import Configuration
from .convert_ops import HorqruxCircuit, convert_block, convert_observable


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

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        ops = convert_block(circuit.block, n_qubits=circuit.n_qubits, config=self.config)
        # hq_circ = jax.jit(HorqruxCircuit(ops).forward)
        return ConvertedCircuit(native=HorqruxCircuit(ops), abstract=circuit, original=circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        hq_obs = convert_observable(observable, n_qubits=n_qubits, config=self.config)
        # comp_circ = jax.jit(hq_obs.forward)
        return ConvertedObservable(native=hq_obs, abstract=observable, original=observable)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: ParamDictType = {},
        state: Any = None,
        endianness: Endianness = Endianness.BIG,
        horqify_state: bool = True,
        unhorqify_state: bool = True,
    ) -> ReturnType:
        n_qubits = circuit.abstract.n_qubits
        param_values = values_to_jax(param_values)
        if state is None:
            state = prepare_state(n_qubits, "0" * n_qubits)
        else:
            state = tensor_to_jnp(pyqify(state)) if horqify_state else state
        state = circuit.native.forward(state, param_values)
        batch_size = 1  # FIXME : add batching
        if unhorqify_state:
            state = jarr_to_tensor(jnp.reshape(state, (batch_size, 2**circuit.abstract.n_qubits)))
        if endianness != self.native_endianness:
            state = jnp.reshape(state, (batch_size, 2**circuit.abstract.n_qubits))
            state = invert_endianness(jarr_to_tensor(state))
        return state

    def run_dm(
        self,
        circuit: ConvertedCircuit,
        noise: Noise,
        param_values: ParamDictType = {},
        state: ReturnType | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> ReturnType:
        raise NotImplementedError

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
        batch_size = infer_batchsize(param_values)
        # TODO vmap circ over batch of values
        n_obs = len(observable)
        if state is None:
            state = prepare_state(circuit.abstract.n_qubits, "0" * circuit.abstract.n_qubits)
        param_values = values_to_jax(param_values)

        def _expectation(state: ArrayLike, param_values: ParamDictType) -> ArrayLike:
            wf = circuit.native.forward(state, param_values)
            return observable[0].native.forward(wf, param_values)
            # return jnp.reshape(exp_vals, (batch_size, n_obs))
        #FIXME reshape, n_obs > 1
        return jax.jit(jax.value_and_grad(_expectation))(state, param_values)

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
        """Samples from a batch of discrete probability distributions.

        Args:
            circuit: A ConvertedCircuit object holding the native PyQ Circuit.
            param_values: A dict holding the embedded parameters which the native ciruit expects.
            n_shots: The number of samples to generate per distribution.
            state: The input state.
            endianness (Endianness): The target endianness of the resulting samples.

        Returns:
            A list of Counter objects where each key represents a bitstring
            and its value the number of times it has been sampled from the given wave function.
        """
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        def _sample(
            _probs: ArrayLike, n_shots: int, endianness: Endianness, n_qubits: int
        ) -> Counter:
            _logits = jax.vmap(lambda _p: jnp.log(_p / (1 - _p)))(_probs)

            def _smple(accumulator: ArrayLike, i: int) -> tuple[ArrayLike, None]:
                accumulator = accumulator.at[i].set(
                    jax.random.categorical(jax.random.PRNGKey(i), _logits)
                )
                return accumulator, None

            samples = jax.lax.scan(
                _smple, jnp.empty_like(jnp.arange(n_shots)), jnp.arange(n_shots)
            )[0]
            return Counter(
                {
                    int_to_basis(k=k, n_qubits=n_qubits, endianness=endianness): count.item()
                    for k, count in enumerate(jnp.bincount(samples))
                    if count > 0
                }
            )

        wf = self.run(
            circuit=circuit,
            param_values=param_values,
            state=state,
            endianness=self.native_endianness,
            horqify_state=True,
            unhorqify_state=False,
        )
        probs = jnp.abs(jnp.float_power(wf, 2.0)).ravel()
        samples = [
            _sample(
                _probs=probs,
                n_shots=n_shots,
                endianness=endianness,
                n_qubits=circuit.abstract.n_qubits,
            ),
        ]
        if endianness != self.native_endianness:
            from qadence.transpile.invert import invert_endianness

            samples = invert_endianness(samples)

        return samples

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: ParamDictType) -> Any:
        raise NotImplementedError

    @staticmethod
    def _overlap(bras: ReturnType, kets: ReturnType) -> ReturnType:
        # TODO
        raise NotImplementedError

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
