from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Tuple

import sympy
import torch
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.tree import Tree

from qadence.blocks.abstract import AbstractBlock
from qadence.parameters import (
    Parameter,
    ParamMap,
    evaluate,
    extract_original_param_entry,
    stringify,
)
from qadence.types import TParameter
from qadence.utils import format_parameter


class PrimitiveBlock(AbstractBlock):
    """
    Primitive blocks represent elementary unitary operations such as single/multi-qubit gates or
    Hamiltonian evolution. See [`qadence.operations`](/qadence/operations.md) for a full list of
    primitive blocks.
    """

    name = "PrimitiveBlock"

    def __init__(self, qubit_support: tuple[int, ...]):
        self._qubit_support = qubit_support

    @property
    def qubit_support(self) -> Tuple[int, ...]:
        return self._qubit_support

    def digital_decomposition(self) -> AbstractBlock:
        """Decomposition into purely digital gates

        This method returns a decomposition of the Block in a
        combination of purely digital single-qubit and two-qubit
        'gates', by manual/custom knowledge of how this can be done efficiently.
        :return:
        """
        return self

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterable:
        yield self

    @property
    def depth(self) -> int:
        return 1

    def __ascii__(self, console: Console) -> RenderableType:
        return Panel(self._block_title, expand=False)

    def __xor__(self, other: int | AbstractBlock) -> AbstractBlock:
        if isinstance(other, int):
            from qadence.transpile import repeat

            B = type(self)
            (start,) = self.qubit_support
            return repeat(B, range(start, start + other))
        else:
            raise TypeError(f"PrimitiveBlocks cannot use ^ on type {type(other)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return self.qubit_support == other.qubit_support
        return False

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "qubit_support": self.qubit_support,
            "tag": self.tag,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> PrimitiveBlock:
        return cls(*d["qubit_support"])

    def __hash__(self) -> int:
        return hash(self._to_json())

    @property
    def n_qubits(self) -> int:
        return max(self.qubit_support) + 1

    @property
    def n_supports(self) -> int:
        return len(self.qubit_support)


class ParametricBlock(PrimitiveBlock):
    """Parameterized primitive blocks"""

    name = "ParametricBlock"

    # a tuple of Parameter's specifies which parameters go into this block
    parameters: ParamMap

    # any unitary can be written as exp(iH).
    # For a parametric block this is particularly interesting and
    # is known for most basic 'gates' or analog pulses.
    generator: AbstractBlock | Parameter | TParameter | None = None

    @property
    def _block_title(self) -> str:
        s = super()._block_title
        params_str = []
        for p in self.parameters.expressions():
            if p.is_number:
                val = evaluate(p)
                if isinstance(val, float):
                    val = round(val, 2)
                params_str.append(val)
            else:
                params_str.append(stringify(p))

        return s + rf" \[params: {params_str}]"

    @property
    def trainable(self) -> bool:
        for expr in self.parameters.expressions():
            if expr.is_number:
                return False
            else:
                return any(not p.trainable for p in expr.free_symbols)
        return True

    @abstractmethod
    def num_parameters(cls) -> int:
        """The number of parameters required by the block

        This is a class property since the number of parameters is defined
        automatically before instantiating the operation. Also, this could
        correspond to a larger number of actual user-facing parameters
        since any parameter expression is allowed

        Examples:
        - RX operation has 1 parameter
        - U operation has 3 parameters
        - HamEvo has 2 parameters (generator and time evolution)
        """
        pass

    def __xor__(self, other: int | AbstractBlock) -> AbstractBlock:
        if isinstance(other, AbstractBlock):
            return super().__xor__(other)
        elif isinstance(other, int):
            from qadence.transpile import repeat

            B = type(self)
            (start,) = self.qubit_support
            (param,) = self.parameters.expressions()
            return repeat(B, range(start, start + other), stringify(param))
        else:
            raise ValueError(f"PrimitiveBlocks cannot use ^ on type {type(other)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return (
                self.qubit_support == other.qubit_support
                and self.parameters.parameter == other.parameters.parameter
            )
        return False

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            raise TypeError(f"Cant check if {type(other)} in {type(self)}")
        for p in self.parameters.expressions():
            if other in p.free_symbols:
                return True
        return False

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "qubit_support": self.qubit_support,
            "tag": self.tag,
            "parameters": self.parameters._to_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> ParametricBlock:
        params = ParamMap._from_dict(d["parameters"])
        target = d["qubit_support"][0]
        return cls(target, params)  # type: ignore[call-arg]

    def dagger(self) -> ParametricBlock:  # type: ignore[override]
        exprs = self.parameters.expressions()
        args = tuple(-extract_original_param_entry(param) for param in exprs)
        args = args if -1 in self.qubit_support else (*self.qubit_support, *args)
        return self.__class__(*args)  # type: ignore[arg-type]


class ScaleBlock(ParametricBlock):
    """Scale blocks are created when multiplying a block by a number or parameter.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import X

    print(X(0) * 2)
    ```
    """

    name = "ScaleBlock"

    block: AbstractBlock

    def __init__(self, block: AbstractBlock, parameter: Any):
        self.block = block
        # TODO: more meaningful name like `scale`?
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        super().__init__(block.qubit_support)

    def __pow__(self, power: int) -> AbstractBlock:
        from qadence.blocks.utils import chain

        expr = self.parameters.parameter
        return ScaleBlock(chain(self.block for _ in range(power)), expr**power)

    @property
    def qubit_support(self) -> Tuple[int, ...]:
        return self.block.qubit_support

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        return self.block.eigenvalues_generator

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.block.eigenvalues

    @property
    def _block_title(self) -> str:
        (scale,) = self.parameters.expressions()
        s = rf"\[mul: {format_parameter(scale)}] "
        return s

    @property
    def n_qubits(self) -> int:
        return self.block.n_qubits

    @property
    def scale(self) -> sympy.Expr:
        (scale,) = self.parameters.expressions()
        return scale

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._block_title)
        else:
            tree = tree.add(self._block_title)
        self.block.__rich_tree__(tree)
        return tree

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        elif isinstance(other, ScaleBlock):
            return (
                self.block == other.block
                and self.parameters.parameter == other.parameters.parameter
            )
        return False

    def __contains__(self, other: object) -> bool:
        from qadence.blocks.composite import CompositeBlock

        if isinstance(other, AbstractBlock):
            if isinstance(self.block, CompositeBlock) and other in self.block:
                return True
            else:
                return self.block == other

        if isinstance(other, Parameter):
            if isinstance(self.block, ParametricBlock) or isinstance(self.block, CompositeBlock):
                return other in self.block
            return False
        else:
            raise TypeError(
                f"Can not check for containment between {type(self)} and {type(other)}."
            )

    def dagger(self) -> ScaleBlock:
        return self.__class__(
            self.block, Parameter(-extract_original_param_entry(self.parameters.parameter))
        )

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "tag": self.tag,
            "parameters": self.parameters._to_dict(),
            "block": self.block._to_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict) -> ScaleBlock:
        from qadence import blocks as qadenceblocks
        from qadence import operations

        expr = ParamMap._from_dict(d["parameters"])
        block: AbstractBlock
        if hasattr(operations, d["block"]["type"]):
            block = getattr(operations, d["block"]["type"])._from_dict(d["block"])

        else:
            block = getattr(qadenceblocks, d["block"]["type"])._from_dict(d["block"])
        return cls(block, expr)  # type: ignore[arg-type]


class TimeEvolutionBlock(ParametricBlock):
    """Simple time evolution block with time-independent Hamiltonian

    This class is just a convenience class which is used to label
    blocks which contains simple time evolution with time-independent
    Hamiltonian operators
    """

    name = "TimeEvolutionBlock"

    @property
    def has_parametrized_generator(self) -> bool:
        return not isinstance(self.generator, AbstractBlock)


class ControlBlock(PrimitiveBlock):
    """The abstract ControlBlock"""

    name = "Control"

    def __init__(self, control: tuple[int, ...], target_block: PrimitiveBlock) -> None:
        self.blocks = (target_block,)

        # using tuple expansion because some control operations could
        # have multiple targets, e.g. CSWAP
        super().__init__((*control, *target_block.qubit_support))  # target_block.qubit_support[0]))

    @property
    def _block_title(self) -> str:
        c, t = self.qubit_support
        s = f"{self.name}({c},{t})"
        return s if self.tag is None else (s + rf" \[tag: {self.tag}]")

    def __ascii__(self, console: Console) -> RenderableType:
        raise NotImplementedError

    @property
    def n_qubits(self) -> int:
        return len(self.qubit_support)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return self.qubit_support == other.qubit_support and self.blocks[0] == other.blocks[0]
        return False

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "qubit_support": self.qubit_support,
            "tag": self.tag,
            "blocks": [b._to_dict() for b in self.blocks],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> ControlBlock:
        control = d["qubit_support"][0]
        target = d["qubit_support"][1]
        return cls(control, target)


class ParametricControlBlock(ParametricBlock):
    """The abstract parametrized ControlBlock"""

    name = "ParameterizedControl"

    def __init__(self, control: tuple[int, ...], target_block: ParametricBlock) -> None:
        self.blocks = (target_block,)
        self.parameters = target_block.parameters
        super().__init__((*control, target_block.qubit_support[0]))

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        return torch.empty(0)

    def __ascii__(self, console: Console) -> RenderableType:
        raise NotImplementedError

    @property
    def n_qubits(self) -> int:
        return len(self.qubit_support)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return self.qubit_support == other.qubit_support and self.blocks[0] == other.blocks[0]
        return False

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "qubit_support": self.qubit_support,
            "tag": self.tag,
            "blocks": [b._to_dict() for b in self.blocks],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> ParametricControlBlock:
        from qadence.serialization import deserialize

        control = d["qubit_support"][0]
        target = d["qubit_support"][1]
        targetblock = d["blocks"][0]
        expr = deserialize(targetblock["parameters"])
        block = cls(control, target, expr)  # type: ignore[call-arg]
        return block
