from __future__ import annotations

import itertools
from abc import abstractproperty
from functools import cached_property, reduce, singledispatch
from typing import Any, ClassVar, Generator, Iterable, Union, cast, get_args

import numpy as np
import torch
from sympy import (
    Basic,
    Expr,
    Float,
    I,
    Integer,
    Matrix,
    Symbol,
)
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.gate import Gate, IdentityGate, UGate

# from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.gate import X as SympyX
from sympy.physics.quantum.gate import Y as SympyY
from sympy.physics.quantum.gate import Z as SympyZ
from torch import Tensor

from qadence.operations import OpEnum
from qadence.types import TNumber


def rebase(args: Iterable) -> Generator:
    """Rebase expression arguments to usable types."""
    # def generate_args() -> Generator:
    # breakpoint()
    for arg in args:
        if isinstance(arg, Mul):
            # Check for negated argument.
            if isinstance(arg.args[0], NegativeOne):
                yield arg
            else:
                yield ChainBlock(*arg.args)
        elif isinstance(arg, Add):
            yield AddBlock(*arg.args)
        elif isinstance(arg, (Integer, Float, Primitive, Composite, Parameter)):
            yield arg


class Parameter(Symbol):
    """Parameters prototype for operation overloadings."""

    def __add__(self, other: Block | Matrix | Basic | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(self, other)

    def __radd__(self, other: Block | Matrix | Basic | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(other, self)

    def __sub__(self, other: Block | Matrix | Basic | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(self, -other)  # type: ignore

    def __rsub__(self, other: Block | Matrix | Basic | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(-self, other)  # type: ignore

    def __mul__(self, other: Block | Matrix | Basic | float | int) -> ChainBlock | Matrix:
        # breakpoint()
        if isinstance(other, Block):
            return ChainBlock(self, other)
        elif isinstance(other, Matrix):
            mul = Mul(self, other)
            return mul.as_mutable()
        elif isinstance(other, (Basic, float, int)):
            return Mul(self, other)
        return ChainBlock(self, other)

    def __rmul__(self, other: Block | Matrix | Basic | float | int) -> ChainBlock | Matrix:
        # breakpoint()
        if isinstance(other, Block):
            return ChainBlock(self, other)
        elif isinstance(other, Matrix):
            mul = Mul(other, self)
            return mul.as_mutable()
        # elif isinstance(other, (Basic, float, int)):
        #     return Mul(other, self)
        return ChainBlock(other, self)


# Should Block(Symbol) or any other Sympy type ?
class Block:
    """
    Define generic composition rules by overloading Python operators
    and symbols for custom block types wrappers around SymPy symbolic
    expressions and additional custom methods and attributes.

    These composition rules/methods are overridden in subtypes if necessary.

    One block to rule them all.
    """

    def __add__(self, other: Block | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(self, other)

    def __radd__(self, other: Block | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(other, self)

    def __sub__(self, other: Block | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(self, -other)  # type: ignore

    def __rsub__(self, other: Block | Mul | float | int) -> AddBlock:
        # breakpoint()
        return AddBlock(-self, other)  # type: ignore

    def __mul__(self, other: Block | float | int) -> ChainBlock:
        # breakpoint()
        return ChainBlock(self, other)

    def __rmul__(self, other: Block | float | int) -> ChainBlock:
        # breakpoint()
        return ChainBlock(other, self)

    def __matmul__(self, other: Block | float | int) -> ChainBlock | KronBlock:
        # breakpoint()
        kron_block = KronBlock(self, other)
        if isinstance(kron_block, Mul):
            return ChainBlock(*kron_block.args)
        return KronBlock(self, other)

    def __rmatmul__(self, other: Block | float | int) -> ChainBlock | KronBlock:
        # breakpoint()
        kron_block = KronBlock(self, other)
        if isinstance(kron_block, Mul):
            return ChainBlock(*kron_block.args)
        return KronBlock(other, self)

    @cached_property
    @abstractproperty
    def arguments(self) -> tuple:
        pass

    @cached_property
    @abstractproperty
    def eigenvalues(self) -> tuple[TNumber, ...]:
        pass

    @cached_property
    @abstractproperty
    def matrix(self) -> Matrix:
        pass

    @cached_property
    @abstractproperty
    def n_qubits(self) -> int:
        pass

    @cached_property
    @abstractproperty
    def parameters(self) -> tuple[TNumber, ...]:
        pass

    @cached_property
    @abstractproperty
    @property
    def qubit_support(self) -> tuple[int, ...]:
        pass


class Composite(Block):
    def __iter__(self) -> Composite:
        self._iterator = iter(self.arguments)
        return self

    def __next__(self) -> Any:
        return next(self._iterator)

    @property
    def arguments(self) -> tuple:
        """Ensure that expression arguments are of the right types."""
        return tuple(rebase(self.args))  # type: ignore

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        return tuple(self.matrix.eigenvals().keys())  # type: ignore

    @property
    def qubit_support(self) -> tuple[int, ...]:
        """Return the qubit support for the corresponding block."""
        # Blocks with no supports (ex: 2.0 * theta as a ChainBlock) returns None.
        arg_supports = tuple(
            set(getattr(arg, "qubit_support")) for arg in self.arguments if isinstance(arg, Block)
        )
        # breakpoint()
        # Filter out None types.
        # arg_supports = tuple(filter(set(), arg_supports))
        if arg_supports:
            return tuple(set.union(*arg_supports))
        return ()

    def dimension(self) -> Generator:
        """
        Dimension block expression arguments to the right support
        by complementing with identities.
        """
        for arg in self.arguments:
            # Complement the expression with identities on disjoint supports.
            # breakpoint()
            if isinstance(arg, Block):
                if arg.qubit_support != self.qubit_support:
                    complement = [
                        Id(qubit) for qubit in set(self.qubit_support) - set(arg.qubit_support)
                    ] + [arg]
                    # If the argument qubit support is not empty,
                    # sort the complementation in-place by qubit-index.
                    # Won't work for multiqubit gates on non-consecutive qubits.
                    if arg.qubit_support:
                        complement.sort(key=lambda x: x.qubit_support[0])
                    arg = kron(*complement)
            elif isinstance(self, AddBlock) and isinstance(arg, (Parameter, Float)):
                id_block = (Id(qubit) for qubit in self.qubit_support)
                kron_id = kron(*id_block)
                arg = arg * kron_id
            elif isinstance(self, AddBlock) and isinstance(arg, Mul):
                chain_block = ChainBlock(*arg.args)
                breakpoint()
                arg = chain_block.dimension()
            yield arg


class AddBlock(Composite, Add):
    """
    A prototype for the AddBlock to define additive composition
    between blocks.
    """

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        # TODO: Take into account multiplicity.
        return tuple(eigenvalue for eigenvalue in rebase(self.matrix.eigenvals().keys()))

    @property
    def matrix(self) -> Matrix:
        # Dimension the expression arguments for matrices to act on the correct subspaces.
        dimensioned_expr = self.dimension()
        # Matriciable arguments are always expected here which is always the case
        # for AddBlock.
        # breakpoint()
        matrices = (getattr(arg, "matrix") for arg in rebase(dimensioned_expr))
        return reduce(lambda x, y: x + y, matrices)

    @property
    def n_qubits(self) -> int:
        # TODO: I don't think traversing the tree to search for the
        # number of qubits is the most efficient way to retrieve it.
        # It should be set when this object is constructed.
        # breakpoint()
        qubit_supports = (
            getattr(arg, "qubit_support") for arg in self.arguments if isinstance(arg, Block)
        )
        qubits = (  # type: ignore
            max(qubit_support) for qubit_support in filter(None, qubit_supports)
        )
        return int(max(qubits)) + 1

    @property
    def parameters(self) -> tuple[TNumber, ...]:
        params = (  # type: ignore
            getattr(arg, "parameters", (arg,))
            for arg in self.arguments
            if not isinstance(arg, Float)
        )
        params = filter(None, params)  # type: ignore
        # breakpoint()
        if params:
            return tuple(itertools.chain(*params))
        return tuple(params)  # type: ignore


add = AddBlock


class ChainBlock(Composite, Mul):
    """
    A prototype for the ChainBlock to define sequential
    composition between blocks.
    """

    def __add__(self, other: Block | float | int) -> AddBlock | ChainBlock:
        # breakpoint()
        if isinstance(self, ChainBlock) and isinstance(other, ChainBlock):
            if self.args[1:] == other.args[1:]:
                mul = AddBlock(self, other)
                return ChainBlock(*mul.args)
            elif self.args[0] == other.args[0]:
                args_block = AddBlock(*self.args[1:], *other.args[1:])
                return ChainBlock(self.args[0], args_block)

        return AddBlock(self, other)

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        return tuple(eigenvalue for eigenvalue in rebase(self.matrix.eigenvals().keys()))

    @property
    def matrix(self) -> Matrix:
        # breakpoint()
        dimensioned_expr = self.dimension()
        matrices = list(getattr(arg, "matrix", arg) for arg in rebase(dimensioned_expr))
        return reduce(lambda x, y: x * y, matrices)

    @property
    def n_qubits(self) -> int:
        # TODO: I don't think traversing the tree to search for the
        # number of qubits is the most efficient way to retrieve it.
        # It should be set when this object is constructed.
        qubits = {
            max(getattr(arg, "qubit_support")) for arg in self.arguments if isinstance(arg, Block)
        }
        return int(max(qubits)) + 1

    @property
    def parameters(self) -> tuple[TNumber, ...]:  # type: ignore
        params = (getattr(arg, "parameters", (arg,)) for arg in self.arguments if not arg.is_Number)
        params = filter(None, params)  # type: ignore
        if params:
            return tuple(itertools.chain(*params))
        return tuple(params)  # type: ignore


chain = ChainBlock


class KronBlock(Composite, TensorProduct):
    """A prototype for the KronBlock to define parallel composition between blocks."""

    def __new__(cls, *args: Block | float | int) -> KronBlock:
        if not args:
            raise NotImplementedError("Can not create an empty KronBlock.")

        qubit_support: set = set()
        for arg in args:
            # breakpoint()
            arg_support = (
                set(getattr(arg, "qubit_support")) if hasattr(arg, "qubit_support") else set()
            )
            if not qubit_support.isdisjoint(arg_support):
                raise ValueError("Input blocks overlap in support.")
            if arg_support is not {}:
                qubit_support.update(arg_support)

        # breakpoint()
        obj: TensorProduct = TensorProduct.__new__(cls, *args)
        return obj  # type: ignore

    @property
    def n_qubits(self) -> int:
        # TODO: I don't think traversing the tree to search for the
        # number of qubits is the most efficient way to retrieve it.
        # It should be set when this object is constructed.
        qubits = {
            max(getattr(arg, "qubit_support")) for arg in self.arguments if isinstance(arg, Block)
        }
        return int(max(qubits)) + 1

    @property
    def matrix(self) -> Matrix:
        # TensorProduct is defined for matriciable arguments only.
        matrices = (getattr(arg, "matrix") for arg in self.arguments if isinstance(arg, Block))
        # breakpoint()
        return reduce(TensorProduct, matrices)

    @property
    def parameters(self) -> tuple[TNumber, ...]:  # type: ignore
        # from qucint.parameters import Parameter

        # params = (getattr(arg, "parameters", (arg,)) for arg in self.arguments)
        params = (getattr(arg, "parameters", (arg,)) for arg in self.arguments if not arg.is_Number)
        params = filter(None, params)  # type: ignore
        if params:
            return tuple(itertools.chain(*params))
        return tuple(params)  # type: ignore


kron = KronBlock


class Primitive(Block, Gate):
    """A prototype for a Primitive block type."""

    def __add__(self, other: Block | float | int) -> AddBlock | ChainBlock:
        # breakpoint()
        if self == other:
            # AddBlock returns a Mul when operands
            # are equal. Convert to ChainBlock.
            mul = AddBlock(self, other)
            return ChainBlock(*mul.args)
        return AddBlock(self, other)

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        return (-1, 1)

    @property
    def matrix(self) -> Matrix:
        return self.get_target_matrix()

    @property
    def n_qubits(self) -> int:
        return int(self.nqubits)

    @property
    def parameters(self) -> tuple[TNumber, ...]:
        return ()

    @property
    def qubit_support(self) -> tuple[int, ...]:
        # breakpoint()
        return cast(tuple, self.targets)


class X(Primitive, SympyX):
    """A prototype for the Pauli X operator."""

    name = OpEnum.x


class Y(Primitive, SympyY):
    """A prototype for the Pauli Y operator."""

    name = OpEnum.y


class Z(Primitive, SympyZ):
    """A prototype for the Pauli Z operator."""

    name = OpEnum.z


class Id(Primitive, IdentityGate):
    """A prototype for the Pauli I operator."""

    name = OpEnum.i

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        return (1, 1)


ArgTypes = Union[Parameter, TNumber, Basic, Expr]


class RX(Primitive, UGate):
    """A prototype for the parametrized RX operation."""

    _generator: ClassVar[Block]
    _parameter: ClassVar[ArgTypes]

    name = OpEnum.rx

    def __new__(cls, *args: ArgTypes) -> RX:
        assert all(isinstance(arg, get_args(ArgTypes)) for arg in args)
        qubit = args[0]
        parameter = args[1]
        generator = X(qubit)
        generator_matrix = generator.matrix * (-I * parameter / 2)
        exp_matrix = generator_matrix.exp()
        # In-place simplification to return a MutableDenseMatrix.
        exp_matrix.simplify()
        obj = UGate.__new__(cls, (qubit,), exp_matrix)
        obj._generator = generator
        obj._parameter = parameter
        return obj  # type: ignore

    @property
    def eigenvalues(self) -> tuple[TNumber, ...]:
        # TODO: Return a more qucint idiomatic type.
        return tuple(self.matrix.eigenvals().keys())

    @property
    def generator(self) -> Block:
        return self._generator

    @property
    def matrix(self) -> Matrix:
        # We need to return the matrix as a mutable object for
        # later in-place operations (Tensorproduct).
        return self.get_target_matrix().as_mutable()

    @property
    def parameters(self) -> tuple[TNumber, ...]:
        if isinstance(self._parameter, get_args(TNumber)):
            return (self._parameter,)
        elif isinstance(self._parameter, (Basic, Expr)):
            return (*self._parameter.free_symbols,)
        return (self._parameter,)


@singledispatch
def evaluate(arg: Any, values: dict = {}) -> Tensor:
    # breakpoint()
    pass


@evaluate.register
def _(arg: Matrix, values: dict = {}) -> Tensor:
    from sympy.matrices import matrix2numpy

    # breakpoint()
    query = {}
    for symbol in arg.free_symbols:
        if symbol.name in values.keys():
            query[symbol] = values[symbol.name]
        elif hasattr(symbol, "value"):
            query[symbol] = symbol.value
        else:
            raise ValueError(f"No value provided for symbol {symbol.name}.")
    res = arg.subs(query)
    return torch.from_numpy(matrix2numpy(res, dtype=np.complex128))


@evaluate.register
def _(arg: tuple, values: dict = {}) -> Tensor:
    # breakpoint()
    res = []
    for expr in arg:
        query = {}
        for symbol in expr.free_symbols:
            if symbol.name in values.keys():
                query[symbol] = values[symbol.name]
            elif hasattr(symbol, "value"):
                query[symbol] = symbol.value
            else:
                raise ValueError(f"No value provided for symbol {symbol.name}.")
            res.append(expr.subs(query))
    # breakpoint()
    return torch.tensor(res, dtype=torch.complex128)
