from __future__ import annotations

from itertools import chain as flatten
from typing import Sequence

import pyqtorch as pyq
import sympy
from torch import (
    Tensor,
    float64,
    tensor,
)
from torch.nn import Module

from qadence.blocks import (
    AbstractBlock,
    AddBlock,
    ChainBlock,
    CompositeBlock,
    MatrixBlock,
    ParametricBlock,
    PrimitiveBlock,
    ScaleBlock,
    TimeEvolutionBlock,
)
from qadence.blocks.primitive import ProjectorBlock
from qadence.operations import (
    U,
    multi_qubit_gateset,
    non_unitary_gateset,
    single_qubit_gateset,
    three_qubit_gateset,
    two_qubit_gateset,
)
from qadence.types import OpName

from .config import Configuration

# Tdagger is not supported currently
supported_gates = list(set(OpName.list()) - set([OpName.TDAGGER]))
"""The set of supported gates.

Tdagger is currently not supported.
"""


def is_single_qubit_chain(block: AbstractBlock) -> bool:
    return (
        isinstance(block, (ChainBlock))
        and block.n_supports == 1
        and all([isinstance(b, (ParametricBlock, PrimitiveBlock)) for b in block])
        and not any([isinstance(b, (ScaleBlock, U)) for b in block])
    )


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = None
) -> Sequence[Module | Tensor | str | sympy.Expr]:
    if isinstance(block, (Tensor, str, sympy.Expr)):  # case for hamevo generators
        if isinstance(block, Tensor):
            block = block.permute(1, 2, 0)  # put batch size in the back
        return [block]
    qubit_support = block.qubit_support
    if n_qubits is None:
        n_qubits = max(qubit_support) + 1

    if config is None:
        config = Configuration()

    if isinstance(block, ScaleBlock):
        scaled_ops = convert_block(block.block, n_qubits, config)
        scale = (
            tensor([block.parameters.parameter], dtype=float64)
            if not block.is_parametric
            else config.get_param_name(block)[0]
        )
        return [pyq.Scale(pyq.Sequence(scaled_ops), scale)]

    elif isinstance(block, TimeEvolutionBlock):
        if isinstance(block, sympy.Symbol):
            generator = block.name  # type: ignore[arg-type]
        else:
            generator = convert_block(block.generator, n_qubits, config)[0]  # type: ignore[arg-type]
        time_param = config.get_param_name(block)[0]
        is_parametric = (
            block.generator.is_parametric if isinstance(block.generator, AbstractBlock) else False
        )
        return [
            pyq.HamiltonianEvolution(
                qubit_support=qubit_support,
                generator=generator,
                time=time_param,
                generator_parametric=is_parametric,  # type: ignore[union-attr]
            )
        ]
    elif isinstance(block, MatrixBlock):
        return [pyq.primitive.Primitive(block.matrix, block.qubit_support)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if isinstance(block, AddBlock):
            return [pyq.Add(ops)]  # add
        elif is_single_qubit_chain(block) and config.use_single_qubit_composition:
            return [pyq.Merge(ops)]  # for chains of single qubit ops on the same qubit
        else:
            return [pyq.Sequence(ops)]  # for kron and chain
    elif isinstance(block, tuple(non_unitary_gateset)):
        if isinstance(block, ProjectorBlock):
            projector = getattr(pyq, block.name)
            if block.name == OpName.N:
                return [projector(target=qubit_support)]
            else:
                return [projector(qubit_support=qubit_support, ket=block.ket, bra=block.bra)]
        else:
            return [getattr(pyq, block.name)(qubit_support[0])]
    elif isinstance(block, tuple(single_qubit_gateset)):
        pyq_cls = getattr(pyq, block.name)
        if isinstance(block, ParametricBlock):
            if isinstance(block, U):
                op = pyq_cls(qubit_support[0], *config.get_param_name(block))
            else:
                op = pyq_cls(qubit_support[0], config.get_param_name(block)[0])
        else:
            op = pyq_cls(qubit_support[0])
        return [op]
    elif isinstance(block, tuple(two_qubit_gateset)):
        pyq_cls = getattr(pyq, block.name)
        if isinstance(block, ParametricBlock):
            op = pyq_cls(qubit_support[0], qubit_support[1], config.get_param_name(block)[0])
        else:
            op = pyq_cls(qubit_support[0], qubit_support[1])
        return [op]
    elif isinstance(block, tuple(three_qubit_gateset) + tuple(multi_qubit_gateset)):
        block_name = block.name[1:] if block.name.startswith("M") else block.name
        pyq_cls = getattr(pyq, block_name)
        if isinstance(block, ParametricBlock):
            op = pyq_cls(qubit_support[:-1], qubit_support[-1], config.get_param_name(block)[0])
        else:
            if "CSWAP" in block_name:
                op = pyq_cls(qubit_support[:-2], qubit_support[-2:])
            else:
                op = pyq_cls(qubit_support[:-1], qubit_support[-1])
        return [op]
    else:
        raise NotImplementedError(
            f"Non supported operation of type {type(block)}. "
            "In case you are trying to run an `AnalogBlock`, make sure you "
            "specify the `device_specs` in your `Register` first."
        )
