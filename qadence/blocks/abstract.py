from __future__ import annotations

import json
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Iterable, Tuple, TypeVar, Union, get_args

import sympy
import torch
from rich.console import Console, RenderableType
from rich.tree import Tree

from qadence.parameters import Parameter
from qadence.types import TNumber


@dataclass(eq=False)  # Avoid unhashability errors due to mutable attributes.
class AbstractBlock(ABC):
    """Base class for both primitive and composite blocks.

    Attributes:
        name (str): A human-readable name attached to the block type. Notice, this is
            the same for all the class instances so it cannot be used for identifying
            different blocks
        qubit_support (tuple[int, ...]): The qubit support of the block expressed as
            a tuple of integers
        tag (str | None): A tag identifying a particular instance of the block which can
            be used for identification and pretty printing
        eigenvalues (list[float] | None): The eigenvalues of the matrix representing the block.
            This is used mainly for primitive blocks and it's needed for generalized parameter
            shift rule computations. Currently unused.
    """

    name: ClassVar[str] = "AbstractBlock"
    tag: str | None = None

    @abstractproperty
    def qubit_support(self) -> Tuple[int, ...]:
        """The indices of the qubit(s) the block is acting on.

        Qadence uses the ordering [0..,N-1] for qubits.
        """
        pass

    @abstractproperty
    def n_qubits(self) -> int:
        """The number of qubits in the whole system.

        A block acting on qubit N would has at least n_qubits >= N + 1.
        """
        pass

    @abstractproperty
    def n_supports(self) -> int:
        """The number of qubits the block is acting on."""
        pass

    @cached_property
    @abstractproperty
    def eigenvalues_generator(self) -> torch.Tensor:
        pass

    @cached_property
    def eigenvalues(
        self, max_num_evals: int | None = None, max_num_gaps: int | None = None
    ) -> torch.Tensor:
        from qadence.utils import eigenvalues

        from .block_to_tensor import block_to_tensor

        return eigenvalues(block_to_tensor(self), max_num_evals, max_num_gaps)

    # make sure that __rmul__ works as expected with np.number
    __array_priority__: int = 1000

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    def __mul__(self, other: Union[AbstractBlock, TNumber, Parameter]) -> AbstractBlock:
        from qadence.blocks.primitive import ScaleBlock
        from qadence.blocks.utils import chain

        # TODO: Improve type checking here
        if isinstance(other, AbstractBlock):
            return chain(self, other)
        else:
            if isinstance(self, ScaleBlock):
                scale = self.parameters.parameter * other
                return ScaleBlock(self.block, scale)
            else:
                scale = Parameter(other)
                return ScaleBlock(self, scale)

    def __rmul__(self, other: AbstractBlock | TNumber | Parameter) -> AbstractBlock:
        return self.__mul__(other)

    def __imul__(self, other: Union[AbstractBlock, TNumber, Parameter]) -> AbstractBlock:
        from qadence.blocks.composite import ChainBlock
        from qadence.blocks.primitive import ScaleBlock
        from qadence.blocks.utils import chain

        if not isinstance(other, AbstractBlock):
            raise TypeError("In-place multiplication is available only for AbstractBlock instances")

        # TODO: Improve type checking here
        if isinstance(other, AbstractBlock):
            return chain(
                *self.blocks if isinstance(self, ChainBlock) else (self,),
                *other.blocks if isinstance(other, ChainBlock) else (other,),
            )
        else:
            if isinstance(self, ScaleBlock):
                p = self.parameters.parameter
                return ScaleBlock(self.block, p * other)
            else:
                return ScaleBlock(self, other if isinstance(other, Parameter) else Parameter(other))

    def __truediv__(self, other: Union[TNumber, sympy.Basic]) -> AbstractBlock:
        if not isinstance(other, (get_args(TNumber), sympy.Basic)):
            raise TypeError("Cannot divide block by another block.")
        ix = 1 / other
        return self * ix  # type: ignore [no-any-return]

    def __add__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.utils import add

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only add a block to another block. Got {type(other)}.")
        return add(self, other)

    def __radd__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.utils import add

        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only add a block to another block. Got {type(other)}.")
        return add(other, self)

    def __iadd__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.composite import AddBlock
        from qadence.blocks.utils import add

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only add a block to another block. Got {type(other)}.")

        # We make sure to unroll any AddBlocks, because for iadd we
        # assume the user expected in-place addition
        return add(
            *self.blocks if isinstance(self, AddBlock) else (self,),
            *other.blocks if isinstance(other, AddBlock) else (other,),
        )

    def __sub__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.primitive import ScaleBlock
        from qadence.blocks.utils import add

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only subtract a block from another block. Got {type(other)}.")
        if isinstance(other, ScaleBlock):
            scale = other.parameters.parameter
            b = ScaleBlock(other.block, -scale)
        else:
            b = ScaleBlock(other, Parameter(-1.0))
        return add(self, b)

    def __isub__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.composite import AddBlock
        from qadence.blocks.utils import add

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only add a block to another block. Got {type(other)}.")

        # We make sure to unroll (and minus) any AddBlocks: for isub we assume the
        # user expected in-place subtraction
        return add(
            *self.blocks if isinstance(self, AddBlock) else (self,),
            *(-block for block in other.blocks) if isinstance(other, AddBlock) else (-other,),
        )

    def __pow__(self, power: int) -> AbstractBlock:
        from qadence.blocks.utils import chain

        return chain(self for _ in range(power))

    def __neg__(self) -> AbstractBlock:
        return self.__mul__(-1.0)

    def __pos__(self) -> AbstractBlock:
        return self

    def __matmul__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.utils import kron

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only kron a block to another block. Got {type(other)}.")
        return kron(self, other)

    def __imatmul__(self, other: AbstractBlock) -> AbstractBlock:
        from qadence.blocks.composite import KronBlock
        from qadence.blocks.utils import kron

        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Can only kron a block with another block. Got {type(other)}.")

        # We make sure to unroll any KronBlocks, because for ixor we assume the user
        # expected in-place kron
        return kron(
            *self.blocks if isinstance(self, KronBlock) else (self,),
            *other.blocks if isinstance(other, KronBlock) else (other,),
        )

    def __iter__(self) -> Iterable:
        yield self

    def __len__(self) -> int:
        return 1

    @property
    def _block_title(self) -> str:
        bits = ",".join(str(i) for i in self.qubit_support)
        s = f"{type(self).__name__}({bits})"

        if self.tag is not None:
            s += rf" \[tag: {self.tag}]"
        return s

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            return Tree(self._block_title)
        else:
            tree.add(self._block_title)
        return tree

    def __repr__(self) -> str:
        console = Console()
        with console.capture() as cap:
            console.print(self.__rich_tree__())
        return cap.get().strip()  # type: ignore [no-any-return]

    @abstractproperty
    def depth(self) -> int:
        pass

    @abstractmethod
    def __ascii__(self, console: Console) -> RenderableType:
        pass

    @abstractmethod
    def _to_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, d: dict) -> AbstractBlock:
        pass

    def _to_json(self) -> str:
        return json.dumps(self._to_dict())

    @classmethod
    def _from_json(cls, path: str | Path) -> AbstractBlock:
        d: dict = dict()
        if isinstance(path, str):
            path = Path(path)
        try:
            with open(path, "r") as file:
                d = json.load(file)

        except Exception as e:
            print(f"Unable to load block due to {e}")

        return AbstractBlock._from_dict(d)

    def _to_file(self, path: str | Path = Path("")) -> None:
        if isinstance(path, str):
            path = Path(path)
        try:
            with open(path, "w") as file:
                file.write(self._to_json())
        except Exception as e:
            print(f"Unable to write {type(self)} to disk due to {e}")

    def __hash__(self) -> int:
        return hash(self._to_json())

    @abstractmethod
    def dagger(self) -> AbstractBlock:
        raise NotImplementedError(
            f"Hermitian adjoint of the Block '{type(self)}' is not implemented yet!"
        )

    @property
    def is_parametric(self) -> bool:
        from qadence.blocks.utils import parameters

        params: list[sympy.Basic] = parameters(self)
        return any(isinstance(p, Parameter) for p in params)

    @property
    def is_time_dependent(self) -> bool:
        from qadence.blocks.utils import parameters

        params: list[sympy.Basic] = parameters(self)
        return any(getattr(p, "is_time", False) for p in params)

    def tensor(self, values: dict[str, TNumber | torch.Tensor] = {}) -> torch.Tensor:
        from .block_to_tensor import block_to_tensor

        return block_to_tensor(self, values)

    @property
    def _is_diag_pauli(self) -> bool:
        from qadence.blocks import CompositeBlock, PrimitiveBlock, ScaleBlock
        from qadence.blocks.utils import block_is_qubit_hamiltonian

        if not block_is_qubit_hamiltonian(self):
            return False

        elif isinstance(self, CompositeBlock):
            return all([b._is_diag_pauli for b in self.blocks])

        elif isinstance(self, ScaleBlock):
            return self.block._is_diag_pauli
        elif isinstance(self, PrimitiveBlock):
            return self.name in ["Z", "I"]
        return False

    @property
    def is_identity(self) -> bool:
        """Identity predicate for blocks."""
        from qadence.blocks import CompositeBlock, PrimitiveBlock, ScaleBlock

        if isinstance(self, CompositeBlock):
            return all([b.is_identity for b in self.blocks])
        elif isinstance(self, ScaleBlock):
            return self.block.is_identity
        elif isinstance(self, PrimitiveBlock):
            return self.name == "I"
        return False


TAbstractBlock = TypeVar("TAbstractBlock", bound=AbstractBlock)
