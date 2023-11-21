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
from jax.typing import ArrayLike

from qadence.backends.utils import tensor_to_jnp
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
from qadence.types import Endianness, OpName, ParamDictType

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
class QdHorQGate(Gate):
    gates: Callable | list[Callable] | Gate | list[Gate]
    target: int | list[int]
    control: int | list[int] | None


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


@register_pytree_node_class
@dataclass
class HorqruxObservable(HorqruxCircuit):
    def __init__(self, operators: list[Gate], n_qubits: int):
        super().__init__(operators=operators)
        self.n_qubits = n_qubits

    def _forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        for op in self.operators:
            state = op.forward(state, values)
        return state

    def forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        return overlap(state, self._forward(state, values))


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration
) -> HorqruxObservable:
    _ops = convert_block(block, n_qubits, config)
    return HorqruxObservable(_ops, n_qubits)  # type: ignore [arg-type]


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


@register_pytree_node_class
class HorqPrimitiveGate(QdHorQGate):
    def __init__(self, gate: Gate, qubit: int):
        self.gates: Gate = gate
        self.target = qubit

    def forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        return apply_gate(state, self.gates(self.target))


@register_pytree_node_class
class HorqCNOTGate(QdHorQGate):
    def __init__(self, gate: Gate, target: int, control: int):
        self.gates: Callable = gate
        self.control: int = control
        self.target: int = target

    def forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        return apply_gate(state, self.gates(self.target, self.control))


@register_pytree_node_class
class HorqKronParametric(QdHorQGate):
    def __init__(self, gates: list[Gate], param_names: list[str], target: list[int]):
        self.gates: list[Gate] = gates
        self.target: list[int] = target
        self.param_names: list[str] = param_names

    def forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        return apply_gate(
            state,
            tuple(
                gate(values[param_name], target)
                for gate, target, param_name in zip(self.gates, self.target, self.param_names)
            ),
        )


@register_pytree_node_class
class HorqKronCNOT(QdHorQGate):
    def __init__(self, gates: list[Gate], target: list[int], control: list[int]):
        self.gates: list[Gate] = gates
        self.target: list[int] = target
        self.control: list[int] = control

    def forward(self, state: ArrayLike, values: ParamDictType) -> ArrayLike:
        return apply_gate(
            state,
            tuple(
                gate(target, control)
                for gate, target, control in zip(self.gates, self.target, self.control)
            ),
        )


@register_pytree_node_class
class HorqParametricGate(QdHorQGate):
    def __init__(self, gate: Gate, qubit: int, parameter_name: str, control: int = None):
        self.gates: Callable = gate
        self.target: int = qubit
        self.parameter: str = parameter_name
        self.control: int | None = control

    def forward(self, state: ArrayLike, values: ParamDictType) -> Array:  # type: ignore [override]
        if isinstance(values, dict):
            val = jnp.array(values[self.parameter])

        else:
            raise

        return apply_gate(state, self.gates(val, self.target, self.control))


@register_pytree_node_class
class HorqAddGate(QdHorQGate):
    def __init__(self, operations: list[QdHorQGate]):
        self.operations = operations

    def forward(self, state: ArrayLike, values: ParamDictType = {}) -> Array:
        return reduce(add, (op.forward(state, values) for op in self.operations))


@register_pytree_node_class
class HorqScaleGate(QdHorQGate):
    def __init__(self, op: QdHorQGate, parameter_name: str):
        self.op: QdHorQGate = op
        self.parameter: str = parameter_name

    def forward(  # type: ignore [override]
        self, state: ArrayLike, values: ParamDictType
    ) -> ArrayLike:
        if isinstance(values, dict):
            val = jnp.array(values[self.parameter])

        else:
            raise

        return val * self.op.forward(state, values)


IMAT = jnp.eye(2, dtype=jnp.cdouble)
ZEROMAT = jnp.zeros_like(IMAT)
XMAT = jnp.array([[0, 1], [1, 0]], dtype=jnp.cdouble)
YMAT = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.cdouble)
ZMAT = jnp.array([[1, 0], [0, -1]], dtype=jnp.cdouble)

MAT_DICT = {"I": IMAT, "Z": ZMAT, "Y": YMAT, "X": XMAT}


def _fill_identities(
    block_mat: Array,
    qubit_support: tuple,
    full_qubit_support: tuple | list,
    diag_only: bool = False,
    endianness: Endianness = Endianness.BIG,
) -> Array:
    qubit_support = tuple(sorted(qubit_support))
    mat = IMAT if qubit_support[0] != full_qubit_support[0] else block_mat
    if diag_only:
        mat = jnp.diag(mat.squeeze(0))
    for i in full_qubit_support[1:]:
        if i == qubit_support[0]:
            other = jnp.diag(block_mat) if diag_only else block_mat
            mat = jnp.kron(mat, other)
        elif i not in qubit_support:
            other = jnp.diag(IMAT) if diag_only else IMAT
            mat = jnp.kron(mat, other)
    return mat


def block_to_jax(
    block: AbstractBlock,
    values: dict = None,
    qubit_support: tuple | None = None,
    use_full_support: bool = True,
    endianness: Endianness = Endianness.BIG,
) -> Array:
    if values is None:
        from qadence.blocks import embedding

        (ps, embed) = embedding(block)
        values = embed(ps, {})

    # get number of qubits
    if qubit_support is None:
        if use_full_support:
            qubit_support = tuple(range(0, block.n_qubits))
        else:
            qubit_support = block.qubit_support
    nqubits = len(qubit_support)

    if isinstance(block, (ChainBlock, KronBlock)):
        # create identity matrix of appropriate dimensions
        mat = IMAT
        for i in range(nqubits - 1):
            mat = jnp.kron(mat, IMAT)

        # perform matrix multiplications
        for b in block.blocks:
            other = block_to_jax(b, values, qubit_support, endianness=endianness)
            mat = jnp.matmul(other, mat)

    elif isinstance(block, AddBlock):
        # create zero matrix of appropriate dimensions
        mat = ZEROMAT
        for _ in range(nqubits - 1):
            mat = jnp.kron(mat, ZEROMAT)

        # perform matrix summation
        for b in block.blocks:
            mat = mat + block_to_jax(b, values, qubit_support, endianness=endianness)

    elif isinstance(block, HorqScaleGate):
        (scale,) = values[block.parameter]
        scale = tensor_to_jnp(scale, dtype=jnp.float64)
        breakpoint()
        mat = scale * block_to_jax(block.block, values, qubit_support, endianness=endianness)

    elif block.name in MAT_DICT.keys():
        block_mat = MAT_DICT[block.name]

        # add missing identities on unused qubits
        mat = _fill_identities(block_mat, block.qubit_support, qubit_support, endianness=endianness)

    else:
        raise TypeError(f"Conversion for block type {type(block)} not supported.")

    return mat
