from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Any, Callable, Dict

import jax.numpy as jnp
from horqrux.analog import _HamiltonianEvolution as NativeHorqHEvo
from horqrux.apply import apply_gates
from horqrux.primitives.parametric import RX, RY, RZ
from horqrux.primitives.primitive import NOT, SWAP, H, I, X, Y, Z
from horqrux.primitives.primitive import Primitive as Gate
from horqrux.composite.sequence import OpSequence as HorqruxSequence
from horqrux.composite.compose import Scale as HorqScaleGate
from horqrux.composite.compose import Add as HorqAddGate
from horqrux.composite.compose import Observable as HorqruxObservable
from horqrux.utils.operator_utils import ControlQubits, TargetQubits
from jax import Array
from jax.scipy.linalg import expm
from jax.tree_util import register_pytree_node_class

from qadence.backends.jax_utils import block_to_jax
from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    CompositeBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
    TimeEvolutionBlock,
)
from qadence.operations import (
    CNOT,
    CRX,
    CRY,
    CRZ,
    MCRX,
    MCRY,
    MCRZ,
    MCZ,
    CZ,
)
from qadence.operations import SWAP as QDSWAP
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
    OpName.CZ: Z,
    OpName.CNOT: NOT,
    OpName.I: I,
    OpName.SWAP: SWAP,
}

supported_gates = list(set(list(ops_map.keys())))


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration
) -> HorqruxObservable:
    ops = convert_block(block, n_qubits, config)
    return HorqruxObservable(ops)


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = Configuration()
) -> list:
    control: ControlQubits
    target: TargetQubits
    if n_qubits is None:
        n_qubits = max(block.qubit_support) + 1
    ops = []
    if isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        ops = [HorqAddGate(ops)] if isinstance(block, AddBlock) else [HorqruxSequence(ops)]
    elif isinstance(block, ScaleBlock):
        op = convert_block(block.block, n_qubits, config=config)[0]
        param_name = config.get_param_name(block)[0]
        ops = [HorqScaleGate(op, param_name)]
    elif block.name in ops_map.keys():
        native_op_fn = ops_map[block.name]
        target, control = (
            (block.qubit_support[1], block.qubit_support[0])
            if isinstance(block, (CZ, CNOT, CRX, CRY, CRZ, QDSWAP))
            else (block.qubit_support[0], (None,))
        )
        native_gate: Gate
        if isinstance(block, ParametricBlock):
            if len(block.parameters._uuid_dict) > 1:
                raise NotImplementedError("Only single-parameter operations are supported.")
            param_name = config.get_param_name(block)[0]
            native_gate = native_op_fn(param=param_name, target=target, control=control)

        elif isinstance(block, PrimitiveBlock):
            if isinstance(block, QDSWAP):
                native_gate = native_op_fn(block.qubit_support[::-1])
            else:
                native_gate = native_op_fn(target=target, control=control)
        ops = [native_gate]

    elif isinstance(block, (MCRX, MCRY, MCRZ, MCZ)):
        block_name = block.name[2:] if block.name.startswith("M") else block.name
        native_op_fn = ops_map[block_name]
        control = block.qubit_support[:-1]
        target = block.qubit_support[-1]

        if isinstance(block, ParametricBlock):
            param = config.get_param_name(block)[0]
            native_gate = native_op_fn(param=param, target=target, control=control)
        else:
            native_gate = native_op_fn(target, control)
        ops = [native_gate]
    elif isinstance(block, TimeEvolutionBlock):
        ops = [HorqHamiltonianEvolution(block, config)]
    else:
        raise NotImplementedError(f"Non-supported operation of type {type(block)}.")

    return ops


@register_pytree_node_class
@dataclass
class HorqHamiltonianEvolution(NativeHorqHEvo):
    def __init__(
        self,
        block: TimeEvolutionBlock,
        config: Configuration,
    ):
        super().__init__("I", block.qubit_support, (None,))
        self._qubit_support = block.qubit_support
        self.param_names = config.get_param_name(block)
        self.block = block
        self.hmat: Array

        if isinstance(block.generator, AbstractBlock) and not block.generator.is_parametric:
            hmat = block_to_jax(
                block.generator,
                qubit_support=self._qubit_support,
                use_full_support=False,
            )
            self.hmat = hmat
            self._hamiltonian = lambda self, values: self.hmat

        else:

            def _hamiltonian(self: HorqHamiltonianEvolution, values: dict[str, Array]) -> Array:
                hmat = block_to_jax(
                    block.generator,  # type: ignore[arg-type]
                    values=values,
                    qubit_support=self._qubit_support,
                    use_full_support=False,
                )
                return hmat

            self._hamiltonian = _hamiltonian

        self._time_evolution = lambda values: values[self.param_names[0]]

    def _unitary(self, values: dict[str, Array]) -> Array:
        """The evolved operator given current parameter values for generator and time evolution."""
        return expm(self._hamiltonian(self, values) * (-1j * self._time_evolution(values)))

    def __call__(
        self,
        state: Array,
        values: dict[str, Array],
    ) -> Array:
        return apply_gates(state, self, values)

    def tree_flatten(self) -> tuple[tuple[NativeHorqHEvo], tuple]:
        children = (self,)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any) -> Any:
        return cls(*children, *aux_data)
