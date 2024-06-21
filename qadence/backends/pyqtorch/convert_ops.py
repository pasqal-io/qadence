from __future__ import annotations

from itertools import chain as flatten
from math import prod
from typing import Any, Sequence, Tuple

import pyqtorch as pyq
import sympy
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import _dagger
from pyqtorch.utils import is_diag
from torch import (
    Tensor,
    cdouble,
    diag_embed,
    diagonal,
    exp,
    float64,
    linalg,
    tensor,
    transpose,
)
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch.nn import Module

from qadence.backends.utils import (
    finitediff,
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
from qadence.blocks.block_to_tensor import _block_to_tensor_embedded, block_to_tensor
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
        # TODO add native pyq hamevo
        # generator = convert_block(block.generator, n_qubits, config)[0]  # type: ignore[arg-type]
        # time_param = config.get_param_name(block)[0]
        # is_parametric = (
        #     block.generator.is_parametric if isinstance(block.generator, AbstractBlock) else False
        # )
        # return [
        #     pyq.HamiltonianEvolution(
        #         qubit_support=qubit_support,
        #         generator=generator,
        #         time=time_param,
        #         generator_parametric=is_parametric,  # type: ignore[union-attr]
        #     )
        # ]
        return [PyQHamiltonianEvolution(qubit_support, n_qubits, block, config)]
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
        self.hmat: Tensor

        if isinstance(block.generator, AbstractBlock) and not block.generator.is_parametric:
            hmat = block_to_tensor(
                block.generator,
                qubit_support=self.qubit_support,
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)
            self.register_buffer("hmat", hmat)
            self._hamiltonian = lambda self, values: self.hmat

        elif isinstance(block.generator, Tensor):
            m = block.generator.to(dtype=cdouble)
            hmat = block_to_tensor(
                MatrixBlock(m, qubit_support=block.qubit_support),
                qubit_support=self.qubit_support,
                use_full_support=False,
            )
            hmat = hmat.permute(1, 2, 0)
            self.register_buffer("hmat", hmat)
            self._hamiltonian = lambda self, values: self.hmat

        elif isinstance(block.generator, sympy.Basic):
            self._hamiltonian = (
                lambda self, values: values[self.param_names[1]].squeeze(3).permute(1, 2, 0)
            )
            # FIXME Why are we squeezing
        else:

            def _hamiltonian(self: PyQHamiltonianEvolution, values: dict[str, Tensor]) -> Tensor:
                hmat = _block_to_tensor_embedded(
                    block.generator,  # type: ignore[arg-type]
                    values=values,
                    qubit_support=self.qubit_support,
                    use_full_support=False,
                    device=self.device,
                )
                return hmat.permute(1, 2, 0)

            self._hamiltonian = _hamiltonian

        self._time_evolution = lambda values: values[self.param_names[0]]
        self._device: torch_device = (
            self.hmat.device if hasattr(self, "hmat") else torch_device("cpu")
        )
        self._dtype: torch_dtype = self.hmat.dtype if hasattr(self, "hmat") else cdouble

    def _unitary(self, hamiltonian: Tensor, time_evolution: Tensor) -> Tensor:
        self.batch_size = max(hamiltonian.size()[2], len(time_evolution))
        diag_check = tensor(
            [is_diag(hamiltonian[..., i]) for i in range(hamiltonian.size()[2])], device=self.device
        )

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
        return self._unitary(self._hamiltonian(self, values), self._time_evolution(values))

    def jacobian_time(self, values: dict[str, Tensor]) -> Tensor:
        """Approximate jacobian of the evolved operator with respect to time evolution."""
        return finitediff(
            lambda t: self._unitary(time_evolution=t, hamiltonian=self._hamiltonian(self, values)),
            values[self.param_names[0]].reshape(-1, 1),
            (0,),
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
                device=self.device,
            )
            return hmat.permute(1, 2, 0)

        return finitediff(
            lambda v: self._unitary(
                time_evolution=self._time_evolution(values), hamiltonian=_generator(v)
            ),
            values[self.param_names[1]].reshape(-1, 1),
            (0,),
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

    @property
    def device(self) -> torch_device:
        return self._device

    @property
    def dtype(self) -> torch_dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> PyQHamiltonianEvolution:
        if hasattr(self, "hmat"):
            self.hmat = self.hmat.to(*args, **kwargs)
            self._device = self.hmat.device
            self._dtype = self.hmat.dtype
        return self
