from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Any, Callable, Dict

import jax.numpy as jnp
from horqrux.abstract import Operator as Gate
from horqrux.apply import apply_gate
from horqrux.parametric import RX, RY, RZ
from horqrux.primitive import NOT, H, I, X, Y, Z
from horqrux.utils import overlap
from jax import Array
from jax.tree_util import register_pytree_node_class

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    CompositeBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
)
from qadence.operations import CNOT, CRX, CRY, CRZ
from qadence.types import OpName, ParamDictType

from .config import Configuration

ops_map: Dict[str, Callable] = {
    OpName.X: X,
    OpName.Y: Y,
    OpName.Z: Z,
    OpName.H: H,
    OpName.RX: RX,
    OpName.RY: RY,
    OpName.RZ: RZ,
    OpName.CRX: RX,
    OpName.CRY: RY,
    OpName.CRZ: RZ,
    OpName.CNOT: NOT,
    OpName.I: I,
}

supported_gates = list(set(list(ops_map.keys())))


@register_pytree_node_class
@dataclass
class HorqruxCircuit:
    operators: list[Gate] = field(default_factory=list)

    def tree_flatten(self) -> tuple[tuple[list[Any]], tuple[()]]:
        children = (self.operators,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)

    def _forward(self, state: Array, values: ParamDictType) -> Array:
        return reduce(lambda state, gate: gate.forward(state, values), self.operators, state)

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return self._forward(state, values)


@register_pytree_node_class
@dataclass
class HorqruxObservable(HorqruxCircuit):
    def __init__(self, operators: list[Gate]):
        super().__init__(operators=operators)

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return overlap(state, self._forward(state, values))


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration
) -> HorqruxObservable:
    ops = convert_block(block, n_qubits, config)
    return HorqruxObservable(ops)


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = Configuration()
) -> list:
    if n_qubits is None:
        n_qubits = max(block.qubit_support) + 1
    ops = []
    if isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        ops = [HorqAddGate(ops)] if isinstance(block, AddBlock) else [HorqruxCircuit(ops)]
    elif isinstance(block, ScaleBlock):
        op = convert_block(block.block, n_qubits, config=config)[0]
        param_name = config.get_param_name(block)[0]
        ops = [HorqScaleGate(op, param_name)]
    elif block.name in ops_map.keys():
        native_op_fn = ops_map[block.name]
        target, control = (
            (block.qubit_support[1], block.qubit_support[0])
            if isinstance(block, (CNOT, CRX, CRY, CRZ))
            else (block.qubit_support[0], (None,))
        )
        native_gate: Gate
        if isinstance(block, ParametricBlock):
            if len(block.parameters._uuid_dict) > 1:
                raise NotImplementedError("Only single-parameter operations are supported.")
            param_name = config.get_param_name(block)[0]
            native_gate = native_op_fn(param=param_name, target=target, control=control)

        elif isinstance(block, PrimitiveBlock):
            native_gate = native_op_fn(target=target, control=control)
        ops = [HorqOperation(native_gate)]

    else:
        raise NotImplementedError(f"Non-supported operation of type {type(block)}.")

    return ops


@register_pytree_node_class
class HorqAddGate(HorqruxCircuit):
    def __init__(self, operations: list[Gate]):
        self.operators = operations
        self.name = "Add"

    def forward(self, state: Array, values: ParamDictType = {}) -> Array:
        return reduce(add, (gate.forward(state, values) for gate in self.operators))

    def __repr__(self) -> str:
        return self.name + f"({self.operators})"


@register_pytree_node_class
@dataclass
class HorqOperation:
    def __init__(self, native_gate: Gate):
        self.native_gate = native_gate

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return apply_gate(state, self.native_gate, values)

    def tree_flatten(self) -> tuple[tuple[Gate], tuple[()]]:
        children = (self.native_gate,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)


@register_pytree_node_class
@dataclass
class HorqScaleGate:
    def __init__(self, gate: HorqOperation, parameter_name: str):
        self.gate = gate
        self.parameter: str = parameter_name

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return jnp.array(values[self.parameter]) * self.gate.forward(state, values)

    def tree_flatten(self) -> tuple[tuple[HorqOperation], tuple[str]]:
        children = (self.gate,)
        aux_data = (self.parameter,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)
