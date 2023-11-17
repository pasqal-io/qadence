from __future__ import annotations

from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Callable, Dict, Union

import jax.numpy as jnp
from horqrux.gates import NOT, H, I, Rx, Ry, Rz, X, Y, Z
from horqrux.ops import apply_gate
from horqrux.types import Gate
from jax import Array
from jax.typing import ArrayLike
from torch import Tensor

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    KronBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
)
from qadence.operations import CNOT, CRX, CRY, CRZ
from qadence.types import OpName

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


class QdHorQGate(Gate):
    gates: Callable | list[Callable] | Gate | list[Gate]
    target: int | list[int]
    control: int | list[int] | None


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = Configuration()
) -> list:
    if n_qubits is None:
        n_qubits = max(block.qubit_support) + 1

    ops: list

    if isinstance(block, AddBlock):
        _ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        ops = [HorqAddGate(_ops)]  # type: ignore [arg-type]

    elif isinstance(block, ChainBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
    elif isinstance(block, KronBlock):
        if all([isinstance(b, ParametricBlock) for b in block.blocks]):
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
            ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))

    elif isinstance(block, CNOT):
        native_op = ops_map[block.name]
        ops = [
            HorqCNOTGate(native_op, block.qubit_support[1], block.qubit_support[0])
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
            )
        ]
    elif isinstance(block, ScaleBlock):
        op = convert_block(block.block, n_qubits, config=config)[0]
        param_name = config.get_param_name(block)[0]
        ops = [HorqScaleGate(op, param_name)]  # type: ignore [list-item, arg-type]

    elif isinstance(block, ParametricBlock):
        native_op = ops_map[block.name]
        # if len(block.parameters) != 1:
        #     raise NotImplementedError("Only 1 parameter operations are implemented")
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
        ops = [HorqPrimitiveGate(gate=native_op, qubit=qubit)]

    else:
        raise NotImplementedError(f"Non supported operation of type {type(block)}")

    return ops  # type: ignore [return-value]


class HorqPrimitiveGate(QdHorQGate):
    def __init__(self, gate: Gate, qubit: int):
        self.gates: Gate = gate
        self.target = qubit

    def forward(self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]) -> ArrayLike:
        return apply_gate(state, self.gates(self.target))


class HorqCNOTGate(QdHorQGate):
    def __init__(self, gate: Gate, target: int, control: int):
        self.gates: Callable = gate
        self.control: int = control
        self.target: int = target

    def forward(self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]) -> ArrayLike:
        return apply_gate(state, self.gates(self.target, self.control))


class HorqKronParametric(QdHorQGate):
    def __init__(self, gates: list[Gate], param_names: list[str], target: list[int]):
        self.gates: list[Gate] = gates
        self.target: list[int] = target
        self.param_names: list[str] = param_names

    def forward(self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]) -> ArrayLike:
        return apply_gate(
            state,
            tuple(
                gate(values[param_name], target)
                for gate, target, param_name in zip(self.gates, self.target, self.param_names)
            ),
        )


class HorqKronCNOT(QdHorQGate):
    def __init__(self, gates: list[Gate], target: list[int], control: list[int]):
        self.gates: list[Gate] = gates
        self.target: list[int] = target
        self.control: list[int] = control

    def forward(self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]) -> ArrayLike:
        return apply_gate(
            state,
            tuple(
                gate(target, control)
                for gate, target, control in zip(self.gates, self.target, self.control)
            ),
        )


class HorqParametricGate(QdHorQGate):
    def __init__(self, gate: Gate, qubit: int, parameter_name: str, control: int = None):
        self.gates: Callable = gate
        self.target: int = qubit
        self.parameter: str = parameter_name
        self.control: int | None = control

    def forward(  # type: ignore [override]
        self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]
    ) -> Array:
        if isinstance(values, dict):
            val = jnp.array(values[self.parameter])

        else:
            raise

        return apply_gate(state, self.gates(val, self.target, self.control))


class HorqAddGate(QdHorQGate):
    def __init__(self, operations: list[QdHorQGate]):
        self.operations = operations

    def forward(self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor] = None) -> Array:
        return reduce(add, (op.forward(state, values) for op in self.operations))


class HorqScaleGate(QdHorQGate):
    def __init__(self, op: QdHorQGate, parameter_name: str):
        self.op: QdHorQGate = op
        self.parameter: str = parameter_name

    def forward(  # type: ignore [override]
        self, state: ArrayLike, values: Union[Dict[str, Tensor], Tensor]
    ) -> ArrayLike:
        if isinstance(values, dict):
            val = jnp.array(values[self.parameter])

        else:
            raise

        return val * self.op.forward(state, values)


class HorQObservable(QdHorQGate):
    def __init__(self, ops: list[QdHorQGate], n_qubits: int):
        self.ops = ops
        self.n_qubits = n_qubits

    def overlap(self, state: ArrayLike, other: ArrayLike) -> ArrayLike:
        batch_size = 1
        state = state.reshape((2**self.n_qubits, batch_size))
        other = other.reshape((2**self.n_qubits, batch_size))
        return jnp.sum(jnp.conj(state) * other).real

    def _forward(self, state: ArrayLike, values: dict) -> ArrayLike:
        for op in self.ops:
            state = op.forward(state, values)
        return state

    def forward(self, state: ArrayLike, values: dict) -> ArrayLike:
        return self.overlap(self._forward(state, values), state)


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration
) -> HorQObservable:
    _ops = convert_block(block, n_qubits, config)
    return HorQObservable(_ops, n_qubits)  # type: ignore [arg-type]
