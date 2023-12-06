from __future__ import annotations

from functools import reduce
from itertools import chain as flatten
from math import prod
from operator import add
from typing import Sequence, Tuple

import pyqtorch as pyq
import sympy
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import _dagger
from pyqtorch.utils import is_diag
from torch import (
    Tensor,
    argsort,
    bmm,
    cdouble,
    diag_embed,
    diagonal,
    exp,
    linalg,
    ones_like,
    permute,
    tensor,
    transpose,
)
from torch.nn import Module

from qadence.backends.utils import (
    finitediff,
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
    _block_to_tensor_embedded,
    block_to_diagonal,
    block_to_tensor,
)
from qadence.blocks.primitive import ProjectorBlock
from qadence.operations import (
    OpName,
    U,
    multi_qubit_gateset,
    non_unitary_gateset,
    single_qubit_gateset,
    three_qubit_gateset,
    two_qubit_gateset,
)
from qadence.utils import infer_batchsize

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


class PyQMatrixBlock(Module):
    def __init__(self, block: MatrixBlock, n_qubits: int, config: Configuration = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.qubits = block.qubit_support
        self.register_buffer("mat", block.matrix.unsqueeze(2))

    def forward(self, state: Tensor, _: dict[str, Tensor] = None) -> Tensor:
        return apply_operator(state, self.mat, self.qubits, self.n_qubits)


class PyQComposedBlock(pyq.QuantumCircuit):
    def __init__(
        self,
        ops: list[Module],
        qubits: Tuple[int, ...],
        n_qubits: int,
        config: Configuration = None,
    ):
        """Compose a chain of single qubit operations on the same qubit into a single.

        call to _apply_batch_gate.
        """
        super().__init__(n_qubits, ops)
        self.qubits = qubits

    def forward(self, state: Tensor, values: dict[str, Tensor] | None = None) -> Tensor:
        batch_size = infer_batchsize(values)
        return apply_operator(
            state, self.unitary(values, batch_size), self.qubits, self.n_qubits, batch_size
        )

    def unitary(self, values: dict[str, Tensor] | None, batch_size: int) -> Tensor:
        batch_first_perm = (2, 0, 1)
        undo_perm = tuple(argsort(tensor(batch_first_perm)))

        def _expand(m: Tensor) -> Tensor:
            if len(m.size()) == 2:
                m = m.unsqueeze(2).repeat(
                    1, 1, batch_size
                )  # Primitive gates are 2D, so we expand them.
            elif m.shape != (2, 2, batch_size):
                m = m.repeat(1, 1, batch_size)  # In case a tensor is 3D doesnt have batch_size.
            return m

        def _batch_first(m: Tensor) -> Tensor:
            return permute(m, batch_first_perm)  # This returns shape (batch_size, 2, 2)

        def _batch_last(m: Tensor) -> Tensor:
            return permute(
                m, undo_perm
            )  # We need to undo the permute since PyQ expects (2, 2, batch_size).

        # We reverse the list of tensors here since matmul is not commutative.

        return _batch_last(
            reduce(
                bmm,
                (_batch_first(_expand(op.unitary(values))) for op in reversed(self.operations)),
            )
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
                return pyqify(diag * unpyqify(state), n_qubits=self.n_qubits)

            self.operation = sparse_operation
        else:
            self.operation = pyq.QuantumCircuit(
                n_qubits,
                convert_block(block, n_qubits, config),
            )

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return pyq.overlap(state, self.operation(state, values))

    def run(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return self.operation(state, values)


class PyQHamiltonianEvolution(Module):
    def __init__(
        self,
        qubit_support: Tuple[int, ...],
        n_qubits: int,
        block: TimeEvolutionBlock,
        config: Configuration,
    ):
        super().__init__()
        self.qubit_support = qubit_support
        self.n_qubits = n_qubits
        self.param_names = config.get_param_name(block)
        self.block = block

        if isinstance(block.generator, AbstractBlock) and not block.generator.is_parametric:
            hmat = block_to_tensor(
                block.generator,
                qubit_support=self.qubit_support,
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)
            self._hamiltonian = lambda x: hmat

        elif isinstance(block.generator, Tensor):
            m = block.generator.to(dtype=cdouble)
            hmat = block_to_tensor(
                MatrixBlock(m, qubit_support=block.qubit_support),
                qubit_support=self.qubit_support,
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)
            self._hamiltonian = lambda x: hmat

        elif isinstance(block.generator, sympy.Basic):
            self._hamiltonian = (
                lambda values: values[self.param_names[1]].squeeze(3).permute(1, 2, 0)
            )
            # FIXME Why are we squeezing
        else:

            def _hamiltonian(values: dict[str, Tensor]) -> Tensor:
                hmat = _block_to_tensor_embedded(
                    block.generator,  # type: ignore[arg-type]
                    values=values,
                    qubit_support=self.qubit_support,
                    use_full_support=False,
                )
                return hmat.permute(1, 2, 0)

            self._hamiltonian = _hamiltonian

        self._time_evolution = lambda values: values[self.param_names[0]]

    def _unitary(self, hamiltonian: Tensor, time_evolution: Tensor) -> Tensor:
        self.batch_size = max(hamiltonian.size()[2], len(time_evolution))
        diag_check = tensor([is_diag(hamiltonian[..., i]) for i in range(hamiltonian.size()[2])])

        def _evolve_diag_operator(hamiltonian: Tensor, time_evolution: Tensor) -> Tensor:
            evol_operator = diagonal(hamiltonian) * (-1j * time_evolution).view((-1, 1))
            evol_operator = diag_embed(exp(evol_operator))
            return transpose(evol_operator, 0, -1)

        def _evolve_matrixexp_operator(hamiltonian: Tensor, time_evolution: Tensor) -> Tensor:
            evol_operator = transpose(hamiltonian, 0, -1) * (-1j * time_evolution).view((-1, 1, 1))
            evol_operator = linalg.matrix_exp(evol_operator)
            return transpose(evol_operator, 0, -1)

        evolve_operator = (
            _evolve_diag_operator if bool(prod(diag_check)) else _evolve_matrixexp_operator
        )
        return evolve_operator(hamiltonian, time_evolution)

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        """The evolved operator given current parameter values for generator and time evolution."""
        return self._unitary(self._hamiltonian(values), self._time_evolution(values))

    def jacobian_time(self, values: dict[str, Tensor]) -> Tensor:
        """Approximate jacobian of the evolved operator with respect to time evolution."""
        return finitediff(
            lambda t: self._unitary(time_evolution=t, hamiltonian=self._hamiltonian(values)),
            values[self.param_names[0]],
        )

    def jacobian_generator(self, values: dict[str, Tensor]) -> Tensor:
        """Approximate jacobian of the evolved operator with respect to generator parameter(s)."""
        if len(self.param_names) > 2:
            raise NotImplementedError(
                "jacobian_generator does not support generators\
                                        with more than 1 parameter."
            )

        def _generator(val: Tensor) -> Tensor:
            val_copy = values.copy()
            val_copy[self.param_names[1]] = val
            hmat = _block_to_tensor_embedded(
                self.block.generator,  # type: ignore[arg-type]
                values=val_copy,
                qubit_support=self.qubit_support,
                use_full_support=False,
            )
            return hmat.permute(1, 2, 0)

        return finitediff(
            lambda v: self._unitary(
                time_evolution=self._time_evolution(values), hamiltonian=_generator(v)
            ),
            values[self.param_names[1]],
        )

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        """Dagger of the evolved operator given the current parameter values."""
        return _dagger(self.unitary(values))

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor],
    ) -> Tensor:
        return apply_operator(
            state,
            self.unitary(values),
            self.qubit_support,
            self.n_qubits,
            self.batch_size,
        )


class AddPyQOperation(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__(n_qubits=n_qubits, operations=operations)

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return reduce(add, (op(state, values) for op in self.operations))


class ScalePyQOperation(pyq.QuantumCircuit):
    def __init__(self, n_qubits: int, block: ScaleBlock, config: Configuration):
        if not isinstance(block.block, PrimitiveBlock):
            raise NotImplementedError(
                "The pyqtorch backend can currently only scale `PrimitiveBlock` types.\
                Please use the following transpile function on your circuit first:\
                from qadence.transpile import scale_primitive_blocks_only"
            )
        ops = convert_block(block.block, n_qubits, config)
        assert len(ops) == 1
        super().__init__(n_qubits, ops)
        (self.param_name,) = config.get_param_name(block)
        self.qubit_support = self.operations[0].qubit_support

    def forward(self, state: Tensor, values: dict[str, Tensor]) -> Tensor:
        return apply_operator(state, self.unitary(values), self.qubit_support, self.n_qubits)

    def unitary(self, values: dict[str, Tensor]) -> Tensor:
        thetas = values[self.param_name]
        return thetas * self.operations[0].unitary(values)

    def dagger(self, values: dict[str, Tensor]) -> Tensor:
        return _dagger(self.unitary(values))

    def jacobian(self, values: dict[str, Tensor]) -> Tensor:
        return values[self.param_name] * ones_like(self.unitary(values))
