from __future__ import annotations

from itertools import chain as flatten
from typing import Callable, Dict, List

from braket.circuits.gates import (
    CZ,
    CNot,
    CPhaseShift,
    CSwap,
    H,
    I,
    Rx,
    Ry,
    Rz,
    S,
    Swap,
    T,
    X,
    Y,
    Z,
)
from braket.circuits.instruction import Instruction
from braket.parametric import FreeParameter

from qadence.blocks import AbstractBlock, CompositeBlock, PrimitiveBlock
from qadence.exceptions import NotSupportedError
from qadence.operations import OpName
from qadence.parameters import evaluate

single_qubit: Dict[str, Callable] = {
    OpName.I: I.i,
    OpName.H: H.h,
    OpName.X: X.x,
    OpName.Y: Y.y,
    OpName.Z: Z.z,
    OpName.S: S.s,
    OpName.T: T.t,
}
single_qubit_parameterized: Dict[str, Callable] = {
    OpName.RX: Rx.rx,
    OpName.RY: Ry.ry,
    OpName.RZ: Rz.rz,
}
two_qubit: Dict[str, Callable] = {OpName.CNOT: CNot.cnot, OpName.SWAP: Swap.swap, OpName.CZ: CZ.cz}
three_qubit: Dict[str, Callable] = {OpName.CSWAP: CSwap.cswap}
two_qubit_parametrized: Dict[str, Callable] = {
    OpName.CPHASE: CPhaseShift.cphaseshift,
}

ops_map = {
    **single_qubit,
    **single_qubit_parameterized,
    **two_qubit,
    **two_qubit_parametrized,
    **three_qubit,
}

supported_gates = list(ops_map.keys())


def BraketOperation(block: PrimitiveBlock) -> Instruction:
    operation = block.name

    if operation in single_qubit:
        return single_qubit[operation](target=block.qubit_support)

    elif operation in single_qubit_parameterized:
        ((uuid, expr),) = block.parameters.items()  # type: ignore [attr-defined]
        if expr.is_number:
            return single_qubit_parameterized[operation](
                target=block.qubit_support, angle=evaluate(expr)  # type: ignore
            )
        else:
            return single_qubit_parameterized[operation](
                target=block.qubit_support,
                angle=FreeParameter(uuid),  # type: ignore
            )

    elif operation in two_qubit:
        return two_qubit[operation](block.qubit_support[0], block.qubit_support[1])

    elif operation in two_qubit_parametrized:
        ((uuid, expr),) = block.parameters.items()  # type: ignore [attr-defined]
        if expr.is_number:
            return two_qubit_parametrized[operation](
                control=block.qubit_support[0],
                target=block.qubit_support[1],
                angle=evaluate(expr),
            )
        else:
            return two_qubit_parametrized[operation](
                control=block.qubit_support[0],
                target=block.qubit_support[1],
                angle=FreeParameter(uuid),
            )
    elif operation in three_qubit:
        return three_qubit[operation](
            block.qubit_support[0], block.qubit_support[1], block.qubit_support[2]
        )
    else:
        raise NotSupportedError(
            "Operation type {} is not supported for Braket backend.".format(type(block))
        )


def convert_block(block: AbstractBlock) -> List[Instruction]:
    if isinstance(block, PrimitiveBlock):
        ops = [BraketOperation(block=block)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(convert_block(b) for b in block.blocks))
    else:
        raise NotSupportedError(
            "Operation type {} is not supported for Braket backend.".format(type(block))
        )
    return ops
