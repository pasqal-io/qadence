from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, fields
from logging import getLogger
from typing import Any, Callable, Iterator, Tuple

from openfermion import QubitOperator
from torch import Tensor
from torch.nn import Module

from qadence.blocks import (
    AbstractBlock,
    CompositeBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
    TimeEvolutionBlock,
    embedding,
)
from qadence.blocks.analog import ConstantAnalogRotation, InteractionBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.noise import NoiseHandler
from qadence.parameters import stringify
from qadence.types import ArrayLike, BackendName, DiffMode, Endianness, Engine, ParamDictType

logger = getLogger(__name__)


@dataclass
class BackendConfiguration:
    _use_gate_params: bool = True
    use_sparse_observable: bool = False
    use_gradient_checkpointing: bool = False
    use_single_qubit_composition: bool = False
    transpilation_passes: list[Callable] | None = None

    def __post_init__(self) -> None:
        if self.transpilation_passes is not None:
            assert all(
                [callable(f) for f in self.transpilation_passes]
            ), "Wrong transpilation passes supplied"
            logger.warning("Custom transpilation passes cannot be serialized in JSON format!")

    def available_options(self) -> str:
        """Return as a string the available fields with types of the configuration.

        Returns:
            str: a string with all the available fields, one per line
        """
        conf_msg = ""
        for _field in fields(self):
            if not _field.name.startswith("_"):
                conf_msg += (
                    f"Name: {_field.name} - Type: {_field.type} - Default value: {_field.default}\n"
                )
        return conf_msg

    @classmethod
    def from_dict(cls, values: dict) -> BackendConfiguration:
        field_names = {field.name for field in fields(cls)}
        init_data = {}

        for key, value in values.items():
            if key not in field_names:
                raise ValueError(f"Unknown field in the configuration: '{key}'.")
            else:
                init_data[key] = value

        return cls(**init_data)

    def get_param_name(self, blk: AbstractBlock) -> Tuple[str, ...]:
        """Return parameter names for the current backend.

        Depending on which backend is in use this
        function returns either UUIDs or expressions of parameters.
        """
        param_ids: Tuple
        # FIXME: better type hiearchy?
        types = (TimeEvolutionBlock, ParametricBlock, ConstantAnalogRotation, InteractionBlock)
        if not isinstance(blk, types):
            raise TypeError(f"Can not infer param name from {type(blk)}")
        else:
            if self._use_gate_params:
                param_ids = tuple(blk.parameters.uuids())
            else:
                param_ids = tuple(map(stringify, blk.parameters.expressions()))
        return param_ids


@dataclass(frozen=True, eq=True)
class Backend(ABC):
    """The abstract class that defines the interface for the backends.

    Attributes:
        name: backend unique string identifier
        supports_ad: whether or not the backend has a native autograd
        supports_bp: whether or not the backend has a native backprop
        supports_adjoint: Does the backend support native adjoint differentation.
        is_remote: whether computations are executed locally or remotely on this
            backend, useful when using cloud platforms where credentials are
            needed for example.
        with_measurements: whether it supports counts or not
        with_noise: whether to add realistic noise or not
        native_endianness: The native endianness of the backend
        engine: The underlying (native) automatic differentiation engine of the backend.
    """

    name: BackendName
    supports_ad: bool
    support_bp: bool
    supports_adjoint: bool
    is_remote: bool
    with_measurements: bool
    native_endianness: Endianness
    engine: Engine

    # FIXME: should this also go into the configuration?
    with_noise: bool

    # additional configuration specific for each backend
    # some backends might not provide any additional configuration
    # but they will still have an empty Configuration class
    config: BackendConfiguration

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            default_conf = self.default_configuration()
            ConfCls = default_conf.__class__

            try:
                new_conf = ConfCls.from_dict(self.config)

                # need this since it is a frozen dataclass
                # see reference documentation
                # https://docs.python.org/3/library/dataclasses.html#frozen-instances
                super().__setattr__("config", new_conf)

            except ValueError as e:
                raise ValueError(f"Wrong configuration provided.\n{str(e)}")

    @abstractmethod
    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        """Converts an abstract `QuantumCircuit` to the native backend representation.

        Arguments:
            circuit: A circuit, for example: `QuantumCircuit(2, X(0))`

        Returns:
            A converted circuit `c`. You can access the original, arbstract circuit via `c.abstract`
            and the converted (or backend *native*) circuit via `c.native`.
        """
        raise NotImplementedError

    @abstractmethod
    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        """Converts an abstract observable (which is just an `AbstractBlock`) to the native backend.

        representation.

        Arguments:
            observable: An observable.
            n_qubits: Number of qubits the observable covers. This is typically `circuit.n_qubits`.

        Returns:
            A converted observable `o`. You can access the original, arbstract observable via
            `o.abstract` and the converted (or backend *native*) observable via `o.native`.
        """
        raise NotImplementedError

    def convert(
        self, circuit: QuantumCircuit, observable: list[AbstractBlock] | AbstractBlock | None = None
    ) -> Converted:
        """Convert an abstract circuit and an optional observable to their native representation.

        Additionally, this function constructs an embedding function which maps from
        user-facing parameters to device parameters (read more on parameter embedding
        [here][qadence.blocks.embedding.embedding]).
        """

        def check_observable(obs_obj: Any) -> AbstractBlock:
            if isinstance(obs_obj, QubitOperator):
                from qadence.blocks.manipulate import from_openfermion

                assert len(obs_obj.terms) > 0, "Make sure to give a non-empty qubit hamiltonian"

                return from_openfermion(obs_obj)

            elif isinstance(obs_obj, (CompositeBlock, PrimitiveBlock, ScaleBlock)):
                from qadence.blocks.utils import block_is_qubit_hamiltonian

                assert block_is_qubit_hamiltonian(
                    obs_obj
                ), "Make sure the QubitHamiltonian consists only of Pauli operators X, Y, Z, I"
                return obs_obj
            raise TypeError(
                "qubit_hamiltonian should be a Pauli-like AbstractBlock or a QubitOperator"
            )

        conv_circ = self.circuit(circuit)
        circ_params, circ_embedding_fn = embedding(
            conv_circ.abstract.block, self.config._use_gate_params, self.engine
        )
        params = circ_params
        if observable is not None:
            observable = observable if isinstance(observable, list) else [observable]
            conv_obs = []
            obs_embedding_fn_list = []

            for obs in observable:
                obs = check_observable(obs)
                c_obs = self.observable(obs, max(circuit.n_qubits, obs.n_qubits))
                obs_params, obs_embedding_fn = embedding(
                    c_obs.abstract, self.config._use_gate_params, self.engine
                )
                params.update(obs_params)
                obs_embedding_fn_list.append(obs_embedding_fn)
                conv_obs.append(c_obs)

            def embedding_fn_dict(a: dict, b: dict) -> dict:
                embedding_dict = circ_embedding_fn(a, b)
                for o in obs_embedding_fn_list:
                    embedding_dict.update(o(a, b))
                return embedding_dict

            return Converted(conv_circ, conv_obs, embedding_fn_dict, params)

        def embedding_fn(a: dict, b: dict) -> dict:
            return circ_embedding_fn(a, b)

        return Converted(conv_circ, None, embedding_fn, params)

    @abstractmethod
    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1000,
        state: ArrayLike | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        """Sample bit strings.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            n_shots: Number of shots to sample.
            state: Initial state.
            noise: A noise model to use.
            mitigation: An error mitigation protocol to apply.
            endianness: Endianness of the resulting bit strings.
        """
        raise NotImplementedError

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, ArrayLike] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
        *args: Any,
        **kwargs: Any,
    ) -> ArrayLike:
        """Run a circuit and return the resulting wave function.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            state: Initial state.
            endianness: Endianness of the resulting wavefunction.

        Returns:
            A list of Counter objects where each key represents a bitstring
            and its value the number of times it has been sampled from the given wave function.
        """
        raise NotImplementedError

    @abstractmethod
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
        """Compute the expectation value of the `circuit` with the given `observable`.

        Arguments:
            circuit: A converted circuit as returned by `backend.circuit`.
            param_values: _**Already embedded**_ parameters of the circuit. See
                [`embedding`][qadence.blocks.embedding.embedding] for more info.
            state: Initial state.
            measurement: Optional measurement protocol. If None, use
                exact expectation value with a statevector simulator.
            noise: A noise model to use.
            endianness: Endianness of the resulting bit strings.
        """
        raise NotImplementedError

    @abstractmethod
    def assign_parameters(self, circuit: ConvertedCircuit, param_values: dict[str, Tensor]) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def default_configuration() -> BackendConfiguration:
        raise NotImplementedError

    def default_diffmode(self) -> DiffMode:
        if self.supports_ad:
            return DiffMode.AD
        else:
            return DiffMode.GPSR


class ConvertedCircuit(Module):
    def __init__(self, native: Any, abstract: QuantumCircuit, original: QuantumCircuit):
        super().__init__()
        self.native = native
        self.abstract = abstract
        self.original = original


class ConvertedObservable(Module):
    def __init__(self, native: Any, abstract: AbstractBlock, original: AbstractBlock):
        super().__init__()
        self.native = native
        self.abstract = abstract
        self.original = original


@dataclass(frozen=True)
class Converted:
    circuit: ConvertedCircuit
    observable: list[ConvertedObservable] | ConvertedObservable | None
    embedding_fn: Callable
    params: ParamDictType

    def __iter__(self) -> Iterator:
        yield self.circuit
        yield self.observable
        yield self.embedding_fn
        yield self.params
