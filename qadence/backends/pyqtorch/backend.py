from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Any

import pyqtorch.modules as pyq
import torch
from torch import Tensor

from qadence.backend import Backend as BackendInterface
from qadence.backend import BackendName, ConvertedCircuit, ConvertedObservable
from qadence.backends.utils import to_list_of_dicts
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.overlap import overlap_exact
from qadence.states import zero_state
from qadence.transpile import (
    add_interaction,
    blockfn_to_circfn,
    chain_single_qubit_ops,
    flatten,
    scale_primitive_blocks_only,
    transpile,
)
from qadence.utils import Endianness, int_to_basis

from .config import Configuration
from .convert_ops import convert_block, convert_observable


@dataclass(frozen=True, eq=True)
class Backend(BackendInterface):
    """PyQTorch backend."""

    # set standard interface parameters
    name: BackendName = BackendName.PYQTORCH
    supports_ad: bool = True
    support_bp: bool = True
    is_remote: bool = False
    with_measurements: bool = True
    with_noise: bool = False
    native_endianness: Endianness = Endianness.BIG
    config: Configuration = Configuration()

    def circuit(self, circuit: QuantumCircuit) -> ConvertedCircuit:
        transpilations = [
            lambda circ: add_interaction(circ, interaction=self.config.interaction),
            lambda circ: blockfn_to_circfn(chain_single_qubit_ops)(circ)
            if self.config.use_single_qubit_composition
            else blockfn_to_circfn(flatten)(circ),
            blockfn_to_circfn(scale_primitive_blocks_only),
        ]

        abstract = transpile(*transpilations)(circuit)  # type: ignore[call-overload]
        ops = convert_block(abstract.block, n_qubits=circuit.n_qubits, config=self.config)
        native = pyq.QuantumCircuit(abstract.n_qubits, ops)
        return ConvertedCircuit(native=native, abstract=abstract, original=circuit)

    def observable(self, observable: AbstractBlock, n_qubits: int) -> ConvertedObservable:
        # make sure only leaves, i.e. primitive blocks are scaled
        transpilations = [
            lambda block: chain_single_qubit_ops(block)
            if self.config.use_single_qubit_composition
            else flatten(block),
            scale_primitive_blocks_only,
        ]
        block = transpile(*transpilations)(observable)  # type: ignore[call-overload]

        (native,) = convert_observable(block, n_qubits=n_qubits, config=self.config)
        return ConvertedObservable(native=native, abstract=block, original=observable)

    def run(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
        pyqify_state: bool = True,
        unpyqify_state: bool = True,
    ) -> Tensor:
        n_qubits = circuit.abstract.n_qubits

        if state is not None:
            if pyqify_state:
                if (state.ndim != 2) or (state.size(1) != 2**n_qubits):
                    raise ValueError(
                        "The initial state must be composed of tensors of size "
                        f"(batch_size, 2**n_qubits). Found: {state.size() = }."
                    )

                # PyQ expects a column vector for the initial state
                # where each element is of dim=2.
                state = state.T.reshape([2] * n_qubits + [state.size(0)])
            else:
                if prod(state.size()[:-1]) != 2**n_qubits:
                    raise ValueError(
                        "A pyqified initial state must be composed of tensors of size "
                        f"(2, 2, ..., batch_size). Found: {state.size() = }."
                    )
        else:
            # infer batch_size without state
            if len(param_values) == 0:
                batch_size = 1
            else:
                batch_size = max([len(tensor) for tensor in param_values.values()])
            state = circuit.native.init_state(batch_size=batch_size)
        state = circuit.native(state, param_values)

        # make sure that the batch dimension is the first one, as standard
        # for PyTorch, and not the last one as done in PyQ
        if unpyqify_state:
            state = torch.flatten(state, start_dim=0, end_dim=-2).t()

        if endianness != self.native_endianness:
            from qadence.transpile import invert_endianness

            state = invert_endianness(state)
        return state

    def _batched_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        protocol: Measurements | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        state = self.run(
            circuit,
            param_values=param_values,
            state=state,
            endianness=endianness,
            pyqify_state=True,
            # we are calling  the native observable directly, so we want to use pyq shapes
            unpyqify_state=False,
        )
        observable = observable if isinstance(observable, list) else [observable]
        _expectation = torch.hstack(
            [obs.native(state, param_values).reshape(-1, 1) for obs in observable]
        )
        return _expectation

    def _looped_expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        protocol: Measurements | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        state = zero_state(circuit.abstract.n_qubits, batch_size=1) if state is None else state
        if state.size(0) != 1:
            raise ValueError(
                "Looping expectation does not make sense with batched initial state. "
                "Define your initial state with `batch_size=1`"
            )

        list_expvals = []
        observables = observable if isinstance(observable, list) else [observable]
        for vals in to_list_of_dicts(param_values):
            wf = self.run(circuit, vals, state, endianness, pyqify_state=True, unpyqify_state=False)
            exs = torch.cat([obs.native(wf, vals) for obs in observables], 0)
            list_expvals.append(exs)

        batch_expvals = torch.vstack(list_expvals)
        return batch_expvals if len(batch_expvals.shape) > 0 else batch_expvals.reshape(1)

    def expectation(
        self,
        circuit: ConvertedCircuit,
        observable: list[ConvertedObservable] | ConvertedObservable,
        param_values: dict[str, Tensor] = {},
        state: Tensor | None = None,
        protocol: Measurements | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        fn = self._looped_expectation if self.config.loop_expectation else self._batched_expectation
        return fn(
            circuit=circuit,
            observable=observable,
            param_values=param_values,
            state=state,
            protocol=protocol,
            endianness=endianness,
        )

    def sample(
        self,
        circuit: ConvertedCircuit,
        param_values: dict[str, Tensor] = {},
        n_shots: int = 1,
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError("You can only call sample with n_shots>0.")

        def _sample(_probs: Tensor, n_shots: int, endianness: Endianness, n_qubits: int) -> Counter:
            return Counter(
                {
                    int_to_basis(k=k, n_qubits=n_qubits, endianness=endianness): count.item()
                    for k, count in enumerate(
                        torch.bincount(
                            torch.multinomial(input=_probs, num_samples=n_shots, replacement=True)
                        )
                    )
                    if count > 0
                }
            )

        wf = self.run(circuit=circuit, param_values=param_values, state=state)
        probs = torch.abs(torch.pow(wf, 2))
        return list(
            map(
                lambda _probs: _sample(
                    _probs=_probs,
                    n_shots=n_shots,
                    endianness=endianness,
                    n_qubits=circuit.abstract.n_qubits,
                ),
                probs,
            )
        )

    def assign_parameters(self, circuit: ConvertedCircuit, param_values: dict[str, Tensor]) -> Any:
        raise NotImplementedError

    @staticmethod
    def _overlap(bras: Tensor, kets: Tensor) -> Tensor:
        return overlap_exact(bras, kets)

    @staticmethod
    def default_configuration() -> Configuration:
        return Configuration()
