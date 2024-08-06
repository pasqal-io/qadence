from __future__ import annotations

from typing import Tuple

import torch
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.tree import Tree

from qadence.parameters import Parameter
from qadence.qubit_support import QubitSupport, QubitSupportType

from .abstract import AbstractBlock
from .primitive import ParametricBlock


class CompositeBlock(AbstractBlock):
    """Block which composes multiple blocks into one larger block (which can again be composed).

    Composite blocks are constructed via [`chain`][qadence.blocks.utils.chain],
    [`kron`][qadence.blocks.utils.kron], and [`add`][qadence.blocks.utils.add].
    """

    name = "CompositeBlock"
    blocks: Tuple[AbstractBlock, ...]

    @property
    def qubit_support(self) -> Tuple[int, ...]:
        from qadence.blocks.analog import AnalogBlock

        anablocks = filter(lambda b: isinstance(b, AnalogBlock), self.blocks)
        digiblocks = filter(lambda b: not isinstance(b, AnalogBlock), self.blocks)
        digital = sum([b.qubit_support for b in digiblocks], start=QubitSupport())
        analog = sum([b.qubit_support for b in anablocks], start=QubitSupport())
        return digital + analog

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        return torch.empty(0)

    @property
    def n_qubits(self) -> int:
        if self.qubit_support:
            return max(self.qubit_support) + 1
        else:
            return 0

    @property
    def n_supports(self) -> int:
        return len(self.qubit_support)

    @property
    def depth(self) -> int:
        return 1 + max([b.depth for b in self.blocks])

    def __iter__(self) -> CompositeBlock:
        self._iterator = iter(self.blocks)
        return self

    def __next__(self) -> AbstractBlock:
        return next(self._iterator)

    def __getitem__(self, item: int) -> AbstractBlock:
        return self.blocks[item]

    def __len__(self) -> int:
        return len(self.blocks)

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._block_title)
        else:
            tree = tree.add(self._block_title)
        for block in self.blocks:
            block.__rich_tree__(tree)
        return tree

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            if len(self.blocks) != len(other.blocks):
                return False
            return self.tag == other.tag and all(
                [b0 == b1 for (b0, b1) in zip(self.blocks, other.blocks)]
            )
        return False

    def __contains__(self, other: object) -> bool:
        # Check containment by instance.
        if isinstance(other, AbstractBlock):
            for b in self.blocks:
                if isinstance(b, CompositeBlock) and other in b:
                    return True
                elif b == other:
                    return True
        elif isinstance(other, Parameter):
            for b in self.blocks:
                if isinstance(b, ParametricBlock) or isinstance(b, CompositeBlock):
                    if other in b:
                        return True
        # Check containment by type.
        elif isinstance(other, type):
            for b in self.blocks:
                if isinstance(b, CompositeBlock) and other in b:
                    return True
                elif type(b) == other:
                    return True
        else:
            raise TypeError(
                f"Can not check for containment between {type(self)} and {type(other)}."
            )
        return False

    def _to_dict(self) -> dict:
        return {
            "type": type(self).__name__,
            "qubit_support": self.qubit_support,
            "tag": self.tag,
            "blocks": [b._to_dict() for b in self.blocks],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> CompositeBlock:
        from qadence import blocks as qadenceblocks
        from qadence import operations
        from qadence.blocks.utils import _construct, tag

        blocks = [
            (
                getattr(operations, b["type"])._from_dict(b)
                if hasattr(operations, b["type"])
                else getattr(qadenceblocks, b["type"])._from_dict(b)
            )
            for b in d["blocks"]
        ]
        block = _construct(cls, blocks)  # type: ignore[arg-type]
        if d["tag"] is not None:
            tag(block, d["tag"])
        return block

    def dagger(self) -> CompositeBlock:  # type: ignore[override]
        reversed_blocks = tuple(block.dagger() for block in reversed(self.blocks))
        return self.__class__(reversed_blocks)  # type: ignore[arg-type]

    def __hash__(self) -> int:
        return hash(self._to_json())


class PutBlock(CompositeBlock):
    name = "put"

    def __init__(self, block: AbstractBlock, support: tuple):
        # np = max(support) + 1 - min(support)
        # nb = block.nqubits
        # assert np == nb, f"You are trying to put a block with {nb} qubits on {np} qubits."
        self.blocks = (block,)
        self._qubit_support = support
        super().__init__()

    @property
    def qubit_support(self) -> Tuple[int, ...]:
        return self._qubit_support

    @property
    def n_qubits(self) -> int:
        return max(self.qubit_support) + 1 - min(self.qubit_support)

    @property
    def _block_title(self) -> str:
        support = ",".join(str(i) for i in self.qubit_support)
        return f"put on ({support})"

    def __ascii__(self, console: Console) -> RenderableType:
        return self.blocks[0].__ascii__(console)

    def dagger(self) -> PutBlock:
        return PutBlock(self.blocks[0].dagger(), self.qubit_support)


class ChainBlock(CompositeBlock):
    """Chains blocks sequentially.

    Constructed via [`chain`][qadence.blocks.utils.chain]
    """

    name = "chain"

    def __init__(self, blocks: Tuple[AbstractBlock, ...]):
        self.blocks = blocks

    def __ascii__(self, console: Console) -> RenderableType:
        # FIXME: deal with other paddings than 1
        padding = 1
        border_width = 1

        # FIXME: deal with primitive block heights other than 3
        h = 3

        def pad(b: AbstractBlock) -> Padding:
            top = (min(b.qubit_support) - min(self.qubit_support)) * h
            return Padding(b.__ascii__(console), (top, 0, 0, 0))

        cols = [pad(b) for b in self.blocks]
        w = sum([console.measure(c).minimum + padding for c in cols])
        w += padding + 2 * border_width

        return Panel(Columns(cols), title=self.tag, width=w)


class KronBlock(CompositeBlock):
    """Stacks blocks horizontally.

    Constructed via [`kron`][qadence.blocks.utils.kron].
    """

    name = "kron"

    def __init__(self, blocks: Tuple[AbstractBlock, ...]):
        if len(blocks) == 0:
            raise NotImplementedError("Empty KronBlocks not supported")

        qubit_support = QubitSupport()
        for b in blocks:
            assert (
                QubitSupportType.GLOBAL,
            ) != b.qubit_support, "Blocks with global support cannot be kron'ed."
            assert qubit_support.is_disjoint(
                b.qubit_support
            ), "Make sure blocks act on distinct qubits!"
            qubit_support += b.qubit_support

        self.blocks = blocks

    def __ascii__(self, console: Console) -> RenderableType:
        ps = [b.__ascii__(console) for b in self.blocks]
        return Panel(Group(*ps), title=self.tag, expand=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, KronBlock):
            if len(self.blocks) != len(other.blocks):
                return False
            return self.tag == other.tag and all([b in other for b in self.blocks])
        return False


class AddBlock(CompositeBlock):
    """Adds blocks.

    Constructed via [`add`][qadence.blocks.utils.add].
    """

    name = "add"

    def __init__(self, blocks: Tuple[AbstractBlock, ...]):
        self.blocks = blocks

    def __ascii__(self, console: Console) -> RenderableType:
        ps = [b.__ascii__(console) for b in self.blocks]
        return Panel(Group(*ps), title=self.tag, expand=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBlock):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, AddBlock):
            if len(self.blocks) != len(other.blocks):
                return False
            return self.tag == other.tag and all([b in other for b in self.blocks])
        return False
