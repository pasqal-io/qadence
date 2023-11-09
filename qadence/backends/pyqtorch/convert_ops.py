from __future__ import annotations

from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Sequence, Tuple

import pyqtorch as pyq
import sympy
from pyqtorch.apply import apply_operator as _apply_batch_gate
from torch import Tensor, argsort, bmm, cdouble, permute, tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint

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
        return [ScalePyQOperation(n_qubits, block, config)]

    elif isinstance(block, AddBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        return [AddPyQOperation(qubit_support, n_qubits, ops, config)]

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

    def forward(self, state: Tensor, _: dict[str, Tensor] = None) -> Tensor:
        return self.apply(self.mat, state)

    def apply(self, matrices: Tensor, state: Tensor) -> Tensor:
        batch_size = state.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)


class PyQComposedBlock(Module):
    def __init__(
        self,
        ops: list[Module],
        qubits: Tuple[int, ...],
        n_qubits: int,
        config: Configuration = None,
    ):
        """Compose a chain of single qubit operations on the same qubit.

        The result is a single call to _apply_batch_gate.
        """
        super().__init__()
        self.operations = ops
        self.qubits = qubits
        self.n_qubits = n_qubits

    def forward(self, state: Tensor, values: dict[str, Tensor] | None = None) -> Tensor:
        batch_size = state.size(-1)
        return self.apply(self.unitary(values, batch_size), state)

    def apply(self, matrices: Tensor, state: Tensor) -> Tensor:
        batch_size = state.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def unitary(self, values: dict[str, Tensor] | None, batch_size: int) -> Tensor:
        perm = (2, 0, 1)  # We permute the dims since bmm expects the batch_dim at 0.

        def _expand_mat(m: Tensor) -> Tensor:
            if len(m.size()) == 2:
                m = m.unsqueeze(2).repeat(
                    1, 1, batch_size
                )  # Primitive gates are 2D, so we expand them.
            elif m.shape != (2, 2, batch_size):
                m = m.repeat(1, 1, batch_size)  # In case a tensor is 3D doesnt have batch_size.
            return permute(m, perm)  # This returns shape (batch_size, 2, 2)

        # We reverse the list of tensors here since matmul is not commutative.
        return permute(
            reduce(bmm, (_expand_mat(op.unitary(values)) for op in reversed(self.operations))),
            tuple(
                argsort(tensor(perm))
            ),  # We need to undo the permute since PyQ expects (2, 2, batch_size).
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

            def sparse_operation(state: Tensor, values: dict[str, Tensor] = None) -> Tensor:
                state = state.reshape(2**self.n_qubits, state.size(-1))
                return (diag * state.T).T

            self.operation = sparse_operation
        else:
            self.operation = pyq.QuantumCircuit(
                n_qubits,
                convert_block(block, n_qubits, config),
            )

        if config.use_gradient_checkpointing:

            def _forward(state: Tensor, values: dict[str, Tensor] = None) -> Tensor:
                new_state = checkpoint(self.operation, state, values, use_reentrant=False)
                return pyq.overlap(state, new_state)

        else:

            def _forward(state: Tensor, values: dict[str, Tensor] = None) -> Tensor:
                return pyq.overlap(state, self.operation(state, values))

        self._forward = _forward

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self._forward(state, values)


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

            def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        elif isinstance(block.generator, Tensor):
            m = block.generator.to(dtype=cdouble)
            hmat = block_to_tensor(
                MatrixBlock(m, qubit_support=block.qubit_support),
                qubit_support=tuple(self.qubits),
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)

            def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        elif isinstance(block.generator, sympy.Basic):

            def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                tevo = values[self.param_names[0]]
                hmat = values[self.param_names[1]]
                hmat = hmat.squeeze(3)  # FIXME: why is this necessary?
                hmat = hmat.permute(1, 2, 0)
                return self.operation(hmat, tevo, state)

        else:

            def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                hmat = _block_to_tensor_embedded(
                    block.generator,  # type: ignore[arg-type]
                    values=values,
                    qubit_support=tuple(self.qubits),
                    use_full_support=False,
                )
                hmat = hmat.permute(1, 2, 0)
                tevo = values[self.param_names[0]]
                return self.operation(hmat, tevo, state)

        if config.use_gradient_checkpointing:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self._forward(state, values)


class AddPyQOperation(Module):
    def __init__(
        self, qubits: Sequence, n_qubits: int, operations: list[Module], config: Configuration
    ):
        super().__init__()
        self.operations = operations

        def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
            return reduce(add, (op(state, values) for op in self.operations))

        if config.use_gradient_checkpointing:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self._forward(state, values)


class ScalePyQOperation(Module):
    """
    Computes:

        M = matrix(op, theta)
        scale * matmul(M, state)
    """

    def __init__(self, n_qubits: int, block: ScaleBlock, config: Configuration):
        super().__init__()
        (self.param_name,) = config.get_param_name(block)
        if not isinstance(block.block, PrimitiveBlock):
            raise NotImplementedError(
                "The pyqtorch backend can currently only scale `PrimitiveBlock` types.\
                Please use the following transpile function on your circuit first:\
                from qadence.transpile import scale_primitive_blocks_only"
            )
        self.operation = convert_block(block.block, n_qubits, config)[0]

        def _fwd(state: Tensor, values: dict[str, Tensor]) -> Tensor:
            return values[self.param_name] * self.operation(state, values)

        if config.use_gradient_checkpointing:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: Tensor, values: dict[str, Tensor]) -> Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param_name]
        return (thetas * self.operation.unitary(values)).unsqueeze(2)

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self._forward(state, values)
