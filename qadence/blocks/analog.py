from __future__ import annotations

from abc import abstractproperty
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from rich.console import Console, RenderableType
from rich.tree import Tree
from sympy import Basic

from qadence.blocks.primitive import AbstractBlock
from qadence.parameters import Parameter, ParamMap, evaluate
from qadence.qubit_support import QubitSupport
from qadence.register import Register
from qadence.types import Interaction


@dataclass(eq=False, repr=False)
class AnalogBlock(AbstractBlock):
    @abstractproperty  # type: ignore[misc, override]
    def qubit_support(self) -> QubitSupport:
        pass

    @abstractproperty
    def duration(self) -> Parameter:
        pass

    def __ascii__(self, console: Console) -> RenderableType:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, d: dict) -> AnalogBlock:
        raise NotImplementedError

    def _to_dict(self) -> dict:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError

    @property
    def n_qubits(self) -> int:
        if self.qubit_support.is_global:
            raise ValueError("Cannot compute number of qubits of a block with global support.")
        return max(self.qubit_support) + 1  # type: ignore[no-any-return]

    @property
    def n_supports(self) -> int:
        if self.qubit_support.is_global:
            raise ValueError("Cannot compute number of qubits of a block with global support.")
        return len(self.qubit_support)  # type: ignore[no-any-return]

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        msg = (
            "Eigenvalues of for generator of analog blocks can be computed via "
            "`add_background_hamiltonian(block, register).eigenvalues_generator`. "
        )
        raise NotImplementedError(msg)

    @property
    def eigenvalues(self) -> torch.Tensor:
        msg = (
            "Eigenvalues of analog blocks can be computed via "
            "`add_background_hamiltonian(block, register).eigenvalues`. "
        )
        raise NotImplementedError(msg)

    @property
    def _block_title(self) -> str:
        t = self.duration
        q = self.qubit_support
        s = f"{type(self).__name__}(t={evaluate(t)}, support={q})"

        if self.tag is not None:
            s += rf" \[tag: {self.tag}]"
        return s

    def compute_eigenvalues_generator(
        self,
        block: AbstractBlock,
        register: Register,
    ) -> torch.Tensor:
        # FIXME: Revisit analog blocks eigenvalues
        from qadence.analog import add_background_hamiltonian

        return add_background_hamiltonian(block, register).eigenvalues_generator  # type: ignore [union-attr]

    def dagger(self) -> AbstractBlock:
        raise NotImplementedError(
            f"Hermitian adjoint of block type {type(self)} is not implemented yet."
        )


@dataclass(eq=False, repr=False)
class InteractionBlock(AnalogBlock):
    """
    Free-evolution for the Hamiltonian interaction term of a register of qubits.

    In real interacting quantum devices, it means letting the system evolve freely according
    to the time-dependent Schrodinger equation. With emulators, this block is translated to an
    appropriate interaction Hamiltonian, for example, an Ising interaction

        Hᵢₙₜ = ∑ᵢⱼ C₆/rᵢⱼ⁶ nᵢnⱼ

    or an XY-interaction

        Hᵢₙₜ = ∑ᵢⱼ C₃/rⱼⱼ³ (XᵢXⱼ + ZᵢZⱼ)

    with `nᵢ = (1-Zᵢ)/2`.

    To construct, use the [`AnalogInteraction`][qadence.operations.AnalogInteraction] function.
    """

    _eigenvalues_generator: torch.Tensor | None = None

    parameters: ParamMap = ParamMap(duration=1000.0)  # ns
    qubit_support: QubitSupport = QubitSupport("global")

    add_pattern: bool = True

    @property
    def eigenvalues_generator(self) -> torch.Tensor | None:
        return self._eigenvalues_generator

    @eigenvalues_generator.setter
    def eigenvalues_generator(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._eigenvalues_generator = value

    @property
    def duration(self) -> Basic:
        return self.parameters.duration


@dataclass(eq=False, repr=False)
class ConstantAnalogRotation(AnalogBlock):
    """Implements a constant analog rotation with interaction dictated by the chosen Hamiltonian.

        H/h = ∑ᵢ(Ω/2 cos(φ)*Xᵢ - sin(φ)*Yᵢ - δnᵢ) + Hᵢₙₜ.

    To construct this block you can use of the following convenience wrappers:
    - The general rotation operation [`AnalogRot`][qadence.operations.AnalogRot]
    - Shorthands for rotatins around an axis:
      [`AnalogRX`][qadence.operations.AnalogRX],
      [`AnalogRY`][qadence.operations.AnalogRY],
      [`AnalogRZ`][qadence.operations.AnalogRZ]

    WARNING: do not use `ConstantAnalogRotation` with `alpha` as differentiable parameter - use
    the convenience wrappers mentioned above.
    """

    _eigenvalues_generator: torch.Tensor | None = None

    parameters: ParamMap = ParamMap(
        alpha=0.0,  # rad
        duration=1000.0,  # ns
        omega=0.0,  # rad/μs
        delta=0.0,  # rad/μs
        phase=0.0,  # rad
    )
    qubit_support: QubitSupport = QubitSupport("global")

    add_pattern: bool = True

    @property
    def _block_title(self) -> str:
        a = self.parameters.alpha
        t = self.parameters.duration
        q = self.qubit_support
        o = self.parameters.omega
        d = self.parameters.delta
        p = self.parameters.phase
        s = f"{type(self).__name__}(α={a}, t={t}, support={q}, Ω={o}, δ={d}, φ={p})"

        if self.tag is not None:
            s += rf" \[tag: {self.tag}]"
        return s

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        if self._eigenvalues_generator is None:
            raise ValueError(
                "Set ConstantAnalogRotation eigenvalues with compute_eigenvalues_generator method."
            )
        return self._eigenvalues_generator

    @eigenvalues_generator.setter
    def eigenvalues_generator(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._eigenvalues_generator = value

    @property
    def duration(self) -> Basic:
        return self.parameters.duration


####################################################################################################


# new, more strict versions of chain/kron blocks to make sure there are no gaps in composed blocks


@dataclass(eq=False, repr=False, init=False)
class AnalogComposite(AnalogBlock):
    blocks: Tuple[AnalogBlock, ...] = ()

    def __init__(self, blocks: Tuple[AnalogBlock, ...]):
        self.blocks = blocks
        # FIXME: add additional InteractionBlock if we have parameterized durations

    @property  # type: ignore[misc, override]
    def qubit_support(self) -> QubitSupport:
        return sum([b.qubit_support for b in self.blocks], start=QubitSupport())

    @abstractproperty
    def duration(self) -> Parameter:
        pass

    @property
    def _block_title(self) -> str:
        t = self.duration
        q = self.qubit_support
        s = f"{type(self).__name__}(t={t}, support={q})"

        if self.tag is not None:
            s += rf" \[tag: {self.tag}]"
        return s

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            tree = Tree(self._block_title)
        else:
            tree = tree.add(self._block_title)
        for block in self.blocks:
            block.__rich_tree__(tree)
        return tree


@dataclass(eq=False, repr=False, init=False)
class AnalogChain(AnalogComposite):
    def __init__(self, blocks: Tuple[AnalogBlock, ...]):
        """A chain of analog blocks.

        Needed because analog blocks require
        stricter validation than the general `ChainBlock`.

        `AnalogChain`s can only be constructed from `AnalogKron` blocks or
        _**globally supported**_, primitive, analog blocks (like `InteractionBlock`s and
        `ConstantAnalogRotation`s).

        Automatically constructed by the [`chain`][qadence.blocks.utils.chain]
        function if only analog blocks are given.

        Example:
        ```python exec="on" source="material-block" result="json"
        from qadence import X, chain, AnalogInteraction

        b = chain(AnalogInteraction(200), AnalogInteraction(200))
        print(type(b))  # this is an `AnalogChain`

        b = chain(X(0), AnalogInteraction(200))
        print(type(b))  # this is a general `ChainBlock`
        ```
        """
        for b in blocks:
            if not (isinstance(b, AnalogKron) or b.qubit_support.is_global):
                raise ValueError("Only KronBlocks or global blocks can be chain'ed.")
        self.blocks = blocks

    @property
    def duration(self) -> Parameter:
        return Parameter(sum(evaluate(b.duration) for b in self.blocks))


@dataclass(eq=False, repr=False, init=False)
class AnalogKron(AnalogComposite):
    def __init__(self, blocks: Tuple[AnalogBlock, ...], interaction: Interaction = Interaction.NN):
        """Stack analog blocks vertically (i.e. in time).

        Needed because analog require
        stricter validation than the general `KronBlock`.

        `AnalogKron`s can only be constructed from _**non-global**_, analog blocks
        with the _**same duration**_.
        """
        if len(blocks) == 0:
            raise NotImplementedError("Empty KronBlocks not supported")

        self.blocks = blocks
        self.interaction = interaction

        qubit_support = QubitSupport()
        duration = blocks[0].duration
        for b in blocks:
            if not isinstance(b, AnalogBlock):
                raise ValueError("Can only kron `AnalgoBlock`s with other `AnalgoBlock`s.")

            if b.qubit_support == QubitSupport("global"):
                raise ValueError("Blocks with global support cannot be kron'ed.")

            if not qubit_support.is_disjoint(b.qubit_support):
                raise ValueError("Make sure blocks act on distinct qubits!")

            if not np.isclose(evaluate(duration), evaluate(b.duration)):
                raise ValueError("Kron'ed blocks have to have same duration.")

            qubit_support += b.qubit_support

        self.blocks = blocks

    @property
    def duration(self) -> Parameter:
        return self.blocks[0].duration


def chain(*args: AnalogBlock) -> AnalogChain:
    return AnalogChain(blocks=args)


def kron(*args: AnalogBlock) -> AnalogKron:
    return AnalogKron(blocks=args)
