from __future__ import annotations

from functools import reduce
from itertools import chain as flatten
from operator import add
from typing import Callable, Sequence

import pyqtorch.modules as pyq
import sympy
import torch
from pyqtorch.core.utils import _apply_batch_gate
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
from qadence.operations import OpName, U

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
    if n_qubits is None:
        n_qubits = max(block.qubit_support) + 1

    if config is None:
        config = Configuration()

    if isinstance(block, ScaleBlock):
        return [ScalePyQOperation(n_qubits, block, config)]

    elif isinstance(block, AddBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        return [AddPyQOperation(block.qubit_support, n_qubits, ops, config)]

    elif isinstance(block, ParametricBlock):
        if isinstance(block, TimeEvolutionBlock):
            op = HEvoPyQOperation(
                qubits=block.qubit_support,
                n_qubits=n_qubits,
                # TODO: use the hevo_algo configuration here to switch between different algorithms
                # for executing the Hamiltonian evolution
                operation=pyq.HamiltonianEvolution(
                    block.qubit_support,
                    n_qubits,
                    n_steps=config.n_steps_hevo,
                ),
                block=block,
                config=config,
            )
        else:
            op = ParametricPyQOperation(n_qubits, block, config)
        return [op]
    elif isinstance(block, MatrixBlock):
        return [PyQMatrixBlock(block, n_qubits, config)]
    elif isinstance(block, PrimitiveBlock):
        return [PyQOperation(n_qubits, block)]
    elif isinstance(block, CompositeBlock):
        ops = list(flatten(*(convert_block(b, n_qubits, config) for b in block.blocks)))
        if is_single_qubit_chain(block) and config.use_single_qubit_composition:
            return [PyQComposedBlock(ops, block.qubit_support, n_qubits, config)]
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

    else:
        msg = (
            f"Non supported operation of type {type(block)}. "
            "In case you are trying to run an `AnalogBlock`, try converting it "
            "with `add_interaction` first."
        )
        raise NotImplementedError(msg)


class PyQOperation(Module):
    def __init__(self, n_qubits: int, block: AbstractBlock):
        super().__init__()
        name = block.name[1:] if block.name.startswith("MC") else block.name
        Op = getattr(pyq, name)
        self.operation = Op(block.qubit_support, n_qubits)

    # primitive blocks do not require any parameter value, hence the
    # second empty argument added here
    def forward(self, state: torch.Tensor, _: dict[str, torch.Tensor] = None) -> torch.Tensor:
        return self.apply(self.matrices(), state)

    def matrices(self, _: dict[str, torch.Tensor] = None) -> torch.Tensor:
        return self.operation.matrix

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.operation.apply(matrices, state)


class ParametricPyQOperation(Module):
    def __init__(self, n_qubits: int, block: ParametricBlock, config: Configuration):
        super().__init__()
        name = block.name[1:] if block.name.startswith("MC") else block.name
        Op = getattr(pyq, name)
        self.operation = Op(block.qubit_support, n_qubits)
        self.param_names = config.get_param_name(block)
        num_params = len(self.param_names)
        if num_params == 1:

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return self.apply(self.matrices(values), state)

        else:

            def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                op_params = {key: values[key] for key in self.param_names}
                max_batch_size = max(p.size() for p in values.values())
                new_values = {
                    k: (v if v.size() == max_batch_size else v.repeat(max_batch_size, 1, 1))
                    for k, v in op_params.items()
                }
                return self.apply(self.matrices(new_values), state)

        if config.use_gradient_checkpointing:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def matrices(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = torch.vstack([values[name] for name in self.param_names])
        return self.operation.matrices(thetas)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.operation.apply(matrices, state)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(state, values)


class PyQMatrixBlock(Module):
    def __init__(self, block: MatrixBlock, n_qubits: int, config: Configuration = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.qubits = block.qubit_support
        self.register_buffer("mat", block.matrix.unsqueeze(2))

    def forward(self, state: torch.Tensor, _: dict[str, torch.Tensor] = None) -> torch.Tensor:
        return self.apply(self.mat, state)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)


class PyQComposedBlock(Module):
    def __init__(
        self,
        ops: list[Module],
        qubits: list[int] | tuple,
        n_qubits: int,
        config: Configuration = None,
    ):
        """Compose a chain of single qubit operations on the same qubit into a single
        call to _apply_batch_gate."""
        super().__init__()
        self.operations = ops
        self.qubits = qubits
        self.n_qubits = n_qubits

    def forward(
        self, state: torch.Tensor, values: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        batch_size = state.size(-1)
        return self.apply(self.matrices(values, batch_size), state)

    def apply(self, matrices: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(-1)
        return _apply_batch_gate(state, matrices, self.qubits, self.n_qubits, batch_size)

    def matrices(self, values: dict[str, torch.Tensor] | None, batch_size: int) -> torch.Tensor:
        perm = (2, 0, 1)  # We permute the dims since torch.bmm expects the batch_dim at 0.

        def _expand_mat(m: torch.Tensor) -> torch.Tensor:
            if len(m.size()) == 2:
                m = m.unsqueeze(2).repeat(
                    1, 1, batch_size
                )  # Primitive gates are 2D, so we expand them.
            elif m.shape != (2, 2, batch_size):
                m = m.repeat(1, 1, batch_size)  # In case a tensor is 3D doesnt have batch_size.
            return torch.permute(m, perm)  # This returns shape (batch_size, 2, 2)

        # We reverse the list of tensors here since matmul is not commutative.
        return torch.permute(
            reduce(
                torch.bmm, (_expand_mat(op.matrices(values)) for op in reversed(self.operations))
            ),
            tuple(
                torch.argsort(torch.tensor(perm))
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

            def sparse_operation(
                state: torch.Tensor, values: dict[str, torch.Tensor] = None
            ) -> torch.Tensor:
                state = state.reshape(2**self.n_qubits, state.size(-1))
                return (diag * state.T).T

            self.operation = sparse_operation
        else:
            self.operation = pyq.QuantumCircuit(
                n_qubits,
                convert_block(block, n_qubits, config),
            )

        if config.use_gradient_checkpointing:

            def _forward(
                state: torch.Tensor, values: dict[str, torch.Tensor] = None
            ) -> torch.Tensor:
                new_state = checkpoint(self.operation, state, values, use_reentrant=False)
                return pyq.overlap(state, new_state)

        else:

            def _forward(
                state: torch.Tensor, values: dict[str, torch.Tensor] = None
            ) -> torch.Tensor:
                return pyq.overlap(state, self.operation(state, values))

        self._forward = _forward

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(state, values)


class HEvoPyQOperation(Module):
    def __init__(
        self,
        qubits: Sequence,
        n_qubits: int,
        operation: Callable,
        block: TimeEvolutionBlock,
        config: Configuration,
    ):
        super().__init__()
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.operation = operation
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

        if config.use_gradient_checkpointing:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(state, values)


class AddPyQOperation(Module):
    def __init__(
        self, qubits: Sequence, n_qubits: int, operations: list[Module], config: Configuration
    ):
        super().__init__()
        self.operations = operations

        def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
            return reduce(add, (op(state, values) for op in self.operations))

        if config.use_gradient_checkpointing:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
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

        def _fwd(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
            return values[self.param_name] * self.operation(state, values)

        if config.use_gradient_checkpointing:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return checkpoint(_fwd, state, values, use_reentrant=False)

        else:

            def _forward(state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
                return _fwd(state, values)

        self._forward = _forward

    def matrices(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        thetas = values[self.param_name]
        return (thetas * self.operation.matrices()).unsqueeze(2)

    def forward(self, state: torch.Tensor, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._forward(state, values)
