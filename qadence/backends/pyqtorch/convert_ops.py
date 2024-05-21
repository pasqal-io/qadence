from __future__ import annotations

from itertools import chain as flatten
from typing import Any, Sequence

import pyqtorch as pyq
from torch import (
    Tensor,
)
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module

from qadence.backends.utils import (
    pyqify,
    unpyqify,
)
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
from qadence.blocks.block_to_tensor import (
    block_to_diagonal,
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


def convert_observable(
    block: AbstractBlock, n_qubits: int, config: Configuration = None
) -> Sequence[Module]:
    return [PyQObservable(block, n_qubits, config)]


def convert_block(
    block: AbstractBlock, n_qubits: int = None, config: Configuration = None
) -> Sequence[Module]:
    qubit_support = block.qubit_support
    if n_qubits is None:
        n_qubits = max(qubit_support) + 1

    if config is None:
        config = Configuration()

    if isinstance(block, ScaleBlock):
        scaled_ops = convert_block(block.block, n_qubits, config)
        return [pyq.Scale(pyq.Sequence(scaled_ops), config.get_param_name(block)[0])]

    elif isinstance(block, TimeEvolutionBlock):
        return [
            pyq.HamiltonianEvolution(
                qubit_support=qubit_support,
                generator=convert_block(block.generator, n_qubits, config)[0],
                time=config.get_param_name(block)[0],
            )
        ]
    elif isinstance(block, MatrixBlock):
        return [pyq.primitive.Primitive(block.matrix, block.qubit_support)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if isinstance(block, AddBlock):
            return [pyq.Add(ops)]
        elif is_single_qubit_chain(block) and config.use_single_qubit_composition:
            return [pyq.Merge(ops)]
        else:
            # NOTE: without wrapping in a pyq.Sequence here the kron/chain
            # blocks won't be properly nested which leads to incorrect results from
            # the `AddBlock`s. For example:
            # add(chain(Z(0), Z(1))) has to result in the following (pseudo-code)
            # AddPyQOperation(pyq.QuantumCircuit(Z, Z))
            # as opposed to
            # AddPyQOperation(Z, Z)
            # which would be wrong.
            return [pyq.Sequence(ops)]
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
            op = pyq_cls(qubit_support[:-1], qubit_support[-1])
        return [op]
    else:
        raise NotImplementedError(
            f"Non supported operation of type {type(block)}. "
            "In case you are trying to run an `AnalogBlock`, make sure you "
            "specify the `device_specs` in your `Register` first."
        )


class PyQObservable(Module):
    def __init__(self, block: AbstractBlock, n_qubits: int, config: Configuration = None):
        super().__init__()
        if config is None:
            config = Configuration()
        self.n_qubits = n_qubits
        if block._is_diag_pauli and not block.is_parametric:
            self.register_buffer("operation", block_to_diagonal(block, tuple(range(n_qubits))))
            self._forward = lambda self, state, values: pyqify(
                self.operation * unpyqify(state), n_qubits=self.n_qubits
            )
        else:
            self.operation = pyq.Sequence(
                convert_block(block, n_qubits, config),
            )
            self._forward = lambda self, state, values: self.operation(state, values)
        self._device = self.operation.device
        self._dtype = self.operation.dtype

    def run(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self._forward(self, state, values)

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return pyq.inner_prod(state, self.run(state, values)).real

    @property
    def device(self) -> torch_device:
        return self._device

    @property
    def dtype(self) -> torch_dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> PyQObservable:
        self.operation = self.operation.to(*args, **kwargs)
        self._device = self.operation.device
        self._dtype = self.operation.dtype
        return self
