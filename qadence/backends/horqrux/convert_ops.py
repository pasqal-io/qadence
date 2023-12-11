from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Any, Callable, Dict

import jax.numpy as jnp
from horqrux.gates import NOT, H, I, Rx, Ry, Rz, X, Y, Z
from horqrux.ops import apply_gate
from horqrux.types import Gate
from horqrux.utils import overlap
from jax import Array
from jax.tree_util import register_pytree_node_class

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    KronBlock,
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
    OpName.RX: Rx,
    OpName.RY: Ry,
    OpName.RZ: Rz,
    OpName.CRX: Rx,
    OpName.CRY: Ry,
    OpName.CRZ: Rz,
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

    def forward(self, state: Array, values: ParamDictType) -> Array:
        for op in self.operators:
            state = op.forward(state, values)
        return state


@dataclass
class HorqruxObservable(HorqruxCircuit):
    def __init__(self, operators: list[Gate]):
        super().__init__(operators=operators)

    def _forward(self, state: Array, values: ParamDictType) -> Array:
        for op in self.operators:
            state = op.forward(state, values)
        return state

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return overlap(state, self._forward(state, values))


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration
) -> HorqruxObservable:
    _ops = convert_block(block, n_qubits, config)
    return HorqruxObservable(_ops)


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = Configuration()
) -> list:
    if n_qubits is None:
        n_qubits = max(block.qubit_support) + 1
    ops = []
    if isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if isinstance(block, AddBlock):
            ops = [HorqAddGate(ops)]
        elif isinstance(block, ChainBlock):
            ops = [HorqruxCircuit(ops)]
        elif isinstance(block, KronBlock):
            if all(
                [
                    isinstance(b, ParametricBlock) and not isinstance(b, ScaleBlock)
                    for b in block.blocks
                ]
            ):
                param_names = [config.get_param_name(b)[0] for b in block.blocks if b.is_parametric]
                ops = [
                    HorqKronParametric(
                        gates=[ops_map[b.name] for b in block.blocks],
                        target=[b.qubit_support[0] for b in block.blocks],
                        param_names=param_names,
                    )
                ]

            elif all([b.name == "CNOT" for b in block.blocks]):
                ops = [
                    HorqKronCNOT(
                        gates=[ops_map[b.name] for b in block.blocks],
                        target=[b.qubit_support[1] for b in block.blocks],
                        control=[b.qubit_support[0] for b in block.blocks],
                    )
                ]
            else:
                ops = [HorqruxCircuit(ops)]

    elif isinstance(block, CNOT):
        native_op = ops_map[block.name]
        ops = [
            HorqCNOTGate(native_op, block.qubit_support[0], block.qubit_support[1])
        ]  # in horqrux target and control are swapped

    elif isinstance(block, (CRX, CRY, CRZ)):
        native_op = ops_map[block.name]
        param_name = config.get_param_name(block)[0]

        ops = [
            HorqParametricGate(
                gate=native_op,
                qubit=block.qubit_support[1],
                parameter_name=param_name,
                control=block.qubit_support[0],
                name=block.name,
            )
        ]
    elif isinstance(block, ScaleBlock):
        op = convert_block(block.block, n_qubits, config=config)[0]
        param_name = config.get_param_name(block)[0]
        ops = [HorqScaleGate(op, param_name)]

    elif isinstance(block, ParametricBlock):
        native_op = ops_map[block.name]
        if len(block.parameters._uuid_dict) > 1:
            raise NotImplementedError("Only single-parameter operations are supported.")
        param_name = config.get_param_name(block)[0]

        ops = [
            HorqParametricGate(
                gate=native_op,
                qubit=block.qubit_support[0],
                parameter_name=param_name,
            )
        ]

    elif isinstance(block, PrimitiveBlock):
        native_op = ops_map[block.name]
        qubit = block.qubit_support[0]
        ops = [HorqPrimitiveGate(gate=native_op, qubit=qubit, name=block.name)]

    else:
        raise NotImplementedError(f"Non-supported operation of type {type(block)}.")

    return ops


class HorqPrimitiveGate:
    def __init__(self, gate: Gate, qubit: int, name: str):
        self.gates: Gate = gate
        self.target = qubit
        self.name = name

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return apply_gate(state, self.gates(self.target))

    def __repr__(self) -> str:
        return self.name + f"(target={self.target})"


class HorqCNOTGate:
    def __init__(self, gate: Gate, control: int, target: int):
        self.gates: Callable = gate
        self.control: int = control
        self.target: int = target

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return apply_gate(state, self.gates(self.target, self.control))


class HorqKronParametric:
    def __init__(self, gates: list[Gate], param_names: list[str], target: list[int]):
        self.operators: list[Gate] = gates
        self.target: list[int] = target
        self.param_names: list[str] = param_names

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return apply_gate(
            state,
            tuple(
                gate(values[param_name], target)
                for gate, target, param_name in zip(self.operators, self.target, self.param_names)
            ),
        )


class HorqKronCNOT(HorqruxCircuit):
    def __init__(self, gates: list[Gate], target: list[int], control: list[int]):
        self.operators: list[Gate] = gates
        self.target: list[int] = target
        self.control: list[int] = control

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return apply_gate(
            state,
            tuple(
                gate(target, control)
                for gate, target, control in zip(self.operators, self.target, self.control)
            ),
        )


class HorqParametricGate:
    def __init__(
        self, gate: Gate, qubit: int, parameter_name: str, control: int = None, name: str = ""
    ):
        self.gates: Callable = gate
        self.target: int = qubit
        self.parameter: str = parameter_name
        self.control: int | None = control
        self.name = name

    def forward(self, state: Array, values: ParamDictType) -> Array:
        val = jnp.array(values[self.parameter])
        return apply_gate(state, self.gates(val, self.target, self.control))

    def __repr__(self) -> str:
        return (
            self.name
            + f"(target={self.target}, parameter={self.parameter}, control={self.control})"
        )


class HorqAddGate(HorqruxCircuit):
    def __init__(self, operations: list[Gate]):
        self.operators = operations
        self.name = "Add"

    def forward(self, state: Array, values: ParamDictType = {}) -> Array:
        return reduce(add, (op.forward(state, values) for op in self.operators))

    def __repr__(self) -> str:
        return self.name + f"({self.operators})"


class HorqScaleGate:
    def __init__(self, op: Gate, parameter_name: str):
        self.op = op
        self.parameter: str = parameter_name

    def forward(self, state: Array, values: ParamDictType) -> Array:
        return jnp.array(values[self.parameter]) * self.op.forward(state, values)
