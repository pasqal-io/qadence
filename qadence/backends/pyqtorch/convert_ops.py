from __future__ import annotations

from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Sequence, Tuple

import pyqtorch as pyq
import sympy
import torch
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import _dagger
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
from qadence.blocks.block_to_tensor import (
    _block_to_tensor_embedded,
    block_to_diagonal,
    block_to_tensor,
)
from qadence.operations import (
    OpName,
    U,
    multi_qubit_gateset,
    non_unitary_gateset,
    single_qubit_gateset,
    three_qubit_gateset,
    two_qubit_gateset,
)

from .config import Configuration

# Tdagger is not supported currently
supported_gates = list(set(OpName.list()) - set([OpName.TDAGGER]))
"""The set of supported gates. Tdagger is currently not supported."""


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
        return [ScalePyQOperation(n_qubits, block, config)]

    elif isinstance(block, AddBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        return [AddPyQOperation(n_qubits, ops)]

    elif isinstance(block, TimeEvolutionBlock):
        return [
            PyQHamiltonianEvolution(
                qubit_support=qubit_support,
                n_qubits=n_qubits,
                block=block,
                config=config,
            )
        ]
    elif isinstance(block, MatrixBlock):
        return [PyQMatrixBlock(block, n_qubits, config)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if is_single_qubit_chain(block) and config.use_single_qubit_composition:
            return [PyQComposedBlock(ops, qubit_support, n_qubits, config)]
        else:
            # NOTE: without wrapping in a pyq.QuantumCircuit here the kron/chain
            # blocks won't be properly nested which leads to incorrect results from
            # the `AddBlock`s. For example:
            # add(chain(Z(0), Z(1))) has to result in the following (pseudo-code)
            # AddPyQOperation(pyq.QuantumCircuit(Z, Z))
            # as opposed to
            # AddPyQOperation(Z, Z)
            # which would be wrong.
            return [pyq.QuantumCircuit(n_qubits, ops)]
    elif isinstance(block, tuple(non_unitary_gateset)):
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
            "In case you are trying to run an `AnalogBlock`, try converting it "
            "with `add_interaction` first."
        )


class PyQMatrixBlock(Module):
    def __init__(self, block: MatrixBlock, n_qubits: int, config: Configuration = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.qubits = block.qubit_support
        self.register_buffer("mat", block.matrix.unsqueeze(2))

    def forward(self, state: torch.Tensor, _: dict[str, torch.Tensor] = None) -> torch.Tensor:
        return apply_operator(state, self.mat, self.qubits, self.n_qubits)


class PyQComposedBlock(pyq.QuantumCircuit):
    def __init__(
        self,
        ops: list[Module],
        qubits: Tuple[int, ...],
        n_qubits: int,
        config: Configuration = None,
    ):
        """Compose a chain of single qubit operations on the same qubit into a single
        call to _apply_batch_gate."""
        super().__init__(n_qubits, ops)
        self.qubits = qubits

    def forward(
        self, state: torch.Tensor, values: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return apply_operator(state, self.unitary(values), self.qubits, self.n_qubits)

    def unitary(self, values: dict[str, torch.Tensor] | None) -> torch.Tensor:
        batch_first_perm = (2, 0, 1)
        undo_perm = tuple(torch.argsort(torch.tensor(batch_first_perm)))
        # We reverse the list of tensors here since matmul is not commutative.
        return torch.permute(
            reduce(
                torch.bmm,
                (
                    torch.permute(op.unitary(values), batch_first_perm)
                    for op in reversed(self.operations)
                ),
            ),
            undo_perm,  # We need to undo the permute since PyQ expects (2, 2, batch_size).
        )


class PyQObservable(Module):
    def __init__(self, block: AbstractBlock, n_qubits: int, config: Configuration = None):
        super().__init__()
        if config is None:
            config = Configuration()
        self.n_qubits = n_qubits
        if block._is_diag_pauli and not block.is_parametric:
            diag = block_to_diagonal(block, tuple(range(n_qubits)))
            self.register_buffer("diag", diag)

            def sparse_operation(
                state: torch.Tensor, values: dict[str, torch.Tensor] = None
            ) -> torch.Tensor:
                state = state.reshape(2**self.n_qubits, state.size(-1))
                return (diag * state.T).T.reshape([2] * self.n_qubits + [state.size(-1)])

            self.operation = sparse_operation
        else:
            self.operation = pyq.QuantumCircuit(
                n_qubits,
                convert_block(block, n_qubits, config),
            )

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return pyq.overlap(state, self.operation(state, values))

    def run(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.operation(state, values)


class PyQHamiltonianEvolution(Module):
    def __init__(
        self,
        qubit_support: Sequence,
        n_qubits: int,
        block: TimeEvolutionBlock,
        config: Configuration,
    ):
        super().__init__()
        self.qubits = qubit_support
        self.n_qubits = n_qubits
        self.operation = pyq.HamiltonianEvolution(qubit_support=qubit_support, n_qubits=n_qubits)
        self.param_names = config.get_param_name(block)
        self._has_parametric_generator: bool
        self.block = block

        if isinstance(block.generator, AbstractBlock) and not block.generator.is_parametric:
            hmat = block_to_tensor(
                block.generator,
                qubit_support=tuple(self.qubits),
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        elif isinstance(block.generator, torch.Tensor):
            m = block.generator.to(dtype=torch.cdouble)
            hmat = block_to_tensor(
                MatrixBlock(m, qubit_support=block.qubit_support),
                qubit_support=tuple(self.qubits),
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        elif isinstance(block.generator, sympy.Basic):

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                tevo = values[self.param_names[0]]
                hmat = values[self.param_names[1]]
                hmat = hmat.squeeze(3)  # FIXME: why is this necessary?
                hmat = hmat.permute(1, 2, 0)
                return self.operation(hmat, tevo, state)

        else:

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                hmat = _block_to_tensor_embedded(
                    block.generator,  # type: ignore[arg-type]
                    values=values,
                    qubit_support=tuple(self.qubits),
                    use_full_support=False,
                )
                hmat = hmat.permute(1, 2, 0)
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        self._forward = _fwd

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(state, values)


class AddPyQOperation(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__(n_qubits=n_qubits, operations=operations)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return reduce(add, (op(state, values) for op in self.operations))


class ScalePyQOperation(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int, block: ScaleBlock, config: Configuration):
        if not isinstance(block.block, PrimitiveBlock):
            raise NotImplementedError(
                "The pyqtorch backend can currently only scale `PrimitiveBlock` types.\
                Please use the following transpile function on your circuit first:\
                from qadence.transpile import scale_primitive_blocks_only"
            )
        super().__init__(n_qubits, convert_block(block.block, n_qubits, config))
        (self.param_name,) = config.get_param_name(block)
        self.qubit_support = self.operations[0].qubit_support

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return apply_operator(state, self.unitary(values), self.qubit_support, self.n_qubits)

    def unitary(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return values[self.param_name] * torch.ones_like(self.unitary(values))
