from __future__ import annotations

from logging import getLogger
from typing import Union

import numpy as np
import torch
from rich.console import Console
from rich.padding import Padding
from torch import Tensor, cdouble, tensor

from qadence.blocks import (
    AbstractBlock,
    PrimitiveBlock,
)
from qadence.blocks.primitive import ProjectorBlock
from qadence.blocks.utils import (
    chain,
    kron,
)
from qadence.noise import NoiseHandler
from qadence.parameters import (
    Parameter,
)
from qadence.types import OpName, TNumber

logger = getLogger(__name__)


class X(PrimitiveBlock):
    """The X gate."""

    name = OpName.X

    def __init__(self, target: int, noise: NoiseHandler | None = None) -> None:
        super().__init__((target,), noise=noise)

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class Y(PrimitiveBlock):
    """The Y gate."""

    name = OpName.Y

    def __init__(self, target: int, noise: NoiseHandler | None = None) -> None:
        super().__init__((target,), noise=noise)

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class Z(PrimitiveBlock):
    """The Z gate."""

    name = OpName.Z

    def __init__(self, target: int, noise: NoiseHandler | None = None) -> None:
        super().__init__((target,), noise=noise)

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class I(PrimitiveBlock):
    """The identity gate."""

    name = OpName.I

    def __init__(self, target: int, noise: NoiseHandler | None = None) -> None:
        super().__init__((target,), noise=noise)

    def __ixor__(self, other: AbstractBlock | int) -> AbstractBlock:
        if not isinstance(other, AbstractBlock):
            raise ValueError(
                f"Can only initialize a kron block with another block. Got {type(other)}."
            )
        return other

    def __imul__(self, other: AbstractBlock | TNumber | Parameter) -> AbstractBlock:
        if not isinstance(other, AbstractBlock):
            raise ValueError(
                "In-place multiplication is available " "only for AbstractBlock instances"
            )
        return other

    @property
    def generator(self) -> AbstractBlock:
        return I(self.qubit_support[0])

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.ones(2, dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.ones(2, dtype=cdouble)

    def __ascii__(self, console: Console) -> Padding:
        return Padding("──────", (1, 1, 1, 1))


class Projector(ProjectorBlock):
    """The projector operator."""

    name = OpName.PROJ

    def __init__(
        self,
        ket: str,
        bra: str,
        qubit_support: int | tuple[int, ...],
        noise: NoiseHandler | None = None,
    ) -> None:
        super().__init__(ket=ket, bra=bra, qubit_support=qubit_support, noise=noise)

    @property
    def generator(self) -> None:
        raise ValueError("Property `generator` not available for non-unitary operator.")

    @property
    def eigenvalues_generator(self) -> None:
        raise ValueError("Property `eigenvalues_generator` not available for non-unitary operator.")


class N(Projector):
    """The N = (1/2)(I-Z) operator."""

    name = OpName.N

    def __init__(
        self,
        target: int,
        state: str = "1",
        noise: NoiseHandler | None = None,
    ) -> None:
        super().__init__(ket=state, bra=state, qubit_support=(target,), noise=noise)

    @property
    def generator(self) -> None:
        raise ValueError("Property `generator` not available for non-unitary operator.")

    @property
    def eigenvalues_generator(self) -> None:
        raise ValueError("Property `eigenvalues_generator` not available for non-unitary operator.")

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([0, 1], dtype=cdouble)


class S(PrimitiveBlock):
    """The S / Phase gate."""

    name = OpName.S

    def __init__(
        self,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = I(target) - Z(target)
        super().__init__((target,), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([1, 1j], dtype=cdouble)

    def dagger(self) -> SDagger:
        return SDagger(self.qubit_support[0])


class SDagger(PrimitiveBlock):
    """The Hermitian adjoint/conjugate transpose of the S / Phase gate."""

    name = OpName.SDAGGER

    def __init__(
        self,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = I(target) - Z(target)
        super().__init__((target,), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([1, -1j], dtype=cdouble)

    def dagger(self) -> S:
        return S(self.qubit_support[0])


class H(PrimitiveBlock):
    """The Hadamard or H gate."""

    name = OpName.H

    def __init__(
        self,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = (1 / np.sqrt(2)) * (X(target) + Z(target) - np.sqrt(2) * I(target))
        super().__init__((target,), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-2, 0], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)


class Zero(PrimitiveBlock):
    name = OpName.ZERO

    def __init__(self) -> None:
        self.generator = 0 * I(0)
        super().__init__((0,))

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.zeros(2, dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.zeros(2, dtype=cdouble)

    def __add__(self, other: AbstractBlock) -> AbstractBlock:
        return other

    def __iadd__(self, other: AbstractBlock) -> AbstractBlock:
        return other

    def __sub__(self, other: AbstractBlock) -> AbstractBlock:
        return -other

    def __isub__(self, other: AbstractBlock) -> AbstractBlock:
        return -other

    def __mul__(self, other: AbstractBlock | TNumber | Parameter) -> AbstractBlock:
        return self

    def __imul__(self, other: AbstractBlock | TNumber | Parameter) -> AbstractBlock:
        return self

    def __ixor__(self, other: AbstractBlock | TNumber | Parameter) -> AbstractBlock:
        return self

    def __pow__(self, power: int) -> AbstractBlock:
        return self


class T(PrimitiveBlock):
    """The T gate."""

    name = OpName.T

    def __init__(
        self,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = I(target) - Z(target)
        super().__init__((target,), noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor([1.0, torch.sqrt(torch.tensor([1j]))], dtype=cdouble)

    @property
    def n_qubits(self) -> int:
        return 1

    def dagger(self) -> TDagger:
        return TDagger(self.qubit_support[0])


class TDagger(PrimitiveBlock):
    """The Hermitian adjoint/conjugate transpose of the T gate."""

    # FIXME: this gate is not support by any backend
    name = "T_dagger"

    def __init__(
        self,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.generator = I(target) - Z(target)
        super().__init__((target,), noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor([1.0, torch.sqrt(torch.tensor([-1j]))], dtype=cdouble)

    @property
    def n_qubits(self) -> int:
        return 1

    def dagger(self) -> T:
        return T(self.qubit_support[0])


class SWAP(PrimitiveBlock):
    """The SWAP gate."""

    name = OpName.SWAP

    def __init__(
        self,
        control: int,
        target: int,
        noise: NoiseHandler | None = None,
    ) -> None:
        a11 = 0.5 * (Z(control) - I(control))
        a22 = -0.5 * (Z(target) + I(target))
        a12 = 0.5 * (chain(X(control), Z(control)) + X(control))
        a21 = 0.5 * (chain(Z(target), X(target)) + X(target))
        self.generator = (
            kron(-1.0 * a22, a11) + kron(-1.0 * a11, a22) + kron(a12, a21) + kron(a21, a12)
        )
        super().__init__((control, target), noise=noise)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-2, 0, 0, 0], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor([-1, 1, 1, 1], dtype=cdouble)

    @property
    def n_qubits(self) -> int:
        return 2

    @property
    def _block_title(self) -> str:
        c, t = self.qubit_support
        s = f"{self.name}({c}, {t})"
        return s if self.tag is None else (s + rf" \[tag: {self.tag}]")


TPauliBlock = Union[X, Y, Z, I, N]
