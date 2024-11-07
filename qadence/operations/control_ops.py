from __future__ import annotations

from logging import getLogger

import sympy
import torch
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.tree import Tree
from torch import Tensor, cdouble

from qadence.blocks import (
    ControlBlock,
    ParametricControlBlock,
)
from qadence.blocks.utils import (
    add,  # noqa
    chain,
    kron,
)
from qadence.noise import NoiseHandler
from qadence.parameters import (
    Parameter,
    evaluate,
)
from qadence.types import OpName, TNumber, TParameter

from .parametric import PHASE, RX, RY, RZ
from .primitive import SWAP, I, N, X, Y, Z

logger = getLogger(__name__)


class CNOT(ControlBlock):
    """The CNot, or CX, gate."""

    name = OpName.CNOT

    def __init__(self, control: int, target: int, noise: NoiseHandler | None = None) -> None:
        self.generator = kron(N(control), X(target) - I(target))
        super().__init__((control,), X(target), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-2, 0, 0, 0], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor([-1, 1, 1, 1], dtype=cdouble)

    def __ascii__(self, console: Console) -> RenderableType:
        (target, control) = self.qubit_support
        h = abs(target - control) + 1
        return Panel(self._block_title, expand=False, height=3 * h)

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            return Tree(self._block_title)
        else:
            tree.add(self._block_title)
        return tree


class MCZ(ControlBlock):
    name = OpName.MCZ

    def __init__(
        self, control: tuple[int, ...], target: int, noise: NoiseHandler | None = None
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target) - I(target))
        super().__init__(control, Z(target), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.cat((torch.tensor(-2, dtype=cdouble), torch.zeros(2**self.n_qubits - 1)))

    @property
    def eigenvalues(self) -> Tensor:
        torch.cat((torch.tensor(-1, dtype=cdouble), torch.ones(2**self.n_qubits - 1)))

    def __ascii__(self, console: Console) -> RenderableType:
        (target, control) = self.qubit_support
        h = abs(target - control) + 1
        return Panel(self._block_title, expand=False, height=3 * h)

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            return Tree(self._block_title)
        else:
            tree.add(self._block_title)
        return tree


class CZ(MCZ):
    """The CZ gate."""

    name = OpName.CZ

    def __init__(self, control: int, target: int, noise: NoiseHandler | None = None) -> None:
        super().__init__((control,), target, noise=noise)


class MCRX(ParametricControlBlock):
    name = OpName.MCRX

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], X(target))
        super().__init__(control, RX(target, parameter), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.cat(
            (torch.zeros(2**self.n_qubits - 2), torch.tensor([1, -1], dtype=cdouble))
        )

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((torch.ones(2**self.n_qubits - 2), lmbd, lmbd.conj()))


class CRX(MCRX):
    """The CRX gate."""

    name = OpName.CRX

    def __init__(
        self,
        control: int,
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ):
        super().__init__((control,), target, parameter, noise=noise)


class MCRY(ParametricControlBlock):
    name = OpName.MCRY

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Y(target))
        super().__init__(control, RY(target, parameter), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.cat(
            (torch.zeros(2**self.n_qubits - 2), torch.tensor([1, -1], dtype=cdouble))
        )

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((torch.ones(2**self.n_qubits - 2), lmbd, lmbd.conj()))


class CRY(MCRY):
    """The CRY gate."""

    name = OpName.CRY

    def __init__(
        self, control: int, target: int, parameter: TParameter, noise: NoiseHandler | None = None
    ):
        super().__init__((control,), target, parameter, noise=noise)


class MCRZ(ParametricControlBlock):
    name = OpName.MCRZ

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target))
        super().__init__(control, RZ(target, parameter), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.cat(
            (torch.zeros(2**self.n_qubits - 2), torch.tensor([1, -1], dtype=cdouble))
        )

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((torch.ones(2**self.n_qubits - 2), lmbd, lmbd.conj()))


class CRZ(MCRZ):
    """The CRZ gate."""

    name = OpName.CRZ

    def __init__(
        self,
        control: int,
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ):
        super().__init__((control,), target, parameter, noise=noise)


class MCPHASE(ParametricControlBlock):
    name = OpName.MCPHASE

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target) - I(target))
        super().__init__(control, PHASE(target, parameter), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.cat(
            (torch.tensor([-2, 0], dtype=cdouble), (torch.zeros(2**self.n_qubits - 2)))
        )

    @property
    def eigenvalues(self) -> Tensor:
        v = evaluate(self.parameters.parameter, as_torch=True)
        return torch.cat((torch.ones(2**self.n_qubits - 1), torch.exp(1j * v)))

    def __rich_tree__(self, tree: Tree = None) -> Tree:
        if tree is None:
            return Tree(self._block_title)
        else:
            tree.add(self._block_title)
        return tree

    def __ascii__(self, console: Console) -> RenderableType:
        (target, control) = self.qubit_support
        h = abs(target - control) + 1
        return Panel(self._block_title, expand=False, height=3 * h)


class CPHASE(MCPHASE):
    """The CPHASE gate."""

    name = OpName.CPHASE

    def __init__(
        self,
        control: int,
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ):
        super().__init__((control,), target, parameter, noise=noise)


class CSWAP(ControlBlock):
    """The CSWAP (Control-SWAP) gate."""

    name = OpName.CSWAP

    def __init__(
        self,
        control: int | tuple[int, ...],
        target1: int,
        target2: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        if isinstance(control, tuple):
            control = control[0]

        a00m = -N(target=control)
        a00p = -N(target=control, state="0")
        a11 = -N(target=target1)
        a22 = -N(target=target2, state="0")
        a12 = 0.5 * (chain(X(target1), Z(target1)) + X(target1))
        a21 = 0.5 * (chain(Z(target2), X(target2)) + X(target2))
        no_effect = kron(a00m, I(target1), I(target2))
        swap_effect = (
            kron(a00p, -1.0 * a22, a11)
            + kron(a00p, -1.0 * a11, a22)
            + kron(a00p, a12, a21)
            + kron(a00p, a21, a12)
        )
        self.generator = no_effect + swap_effect
        super().__init__((control,), SWAP(target1, target2), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor((1, -1, 1, 1, 1, 1, 1, 1), dtype=torch.cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor((1, -1, 1, 1, 1, 1, 1, 1), dtype=torch.cdouble)

    @property
    def nqubits(self) -> int:
        return 3


class Toffoli(ControlBlock):
    name = OpName.TOFFOLI

    def __init__(
        self, control: tuple[int, ...], target: int, noise: NoiseHandler | None = None
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], X(target) - I(target))
        super().__init__(control, X(target), noise=noise)

    @property
    def n_qubits(self) -> int:
        return len(self.qubit_support)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor(
            [-2, *[0 for _ in range(2 ** len(self.qubit_support) - 1)]], dtype=cdouble
        )

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor(
            [-1, *[1 for _ in range(2 ** len(self.qubit_support) - 1)]], dtype=cdouble
        )
