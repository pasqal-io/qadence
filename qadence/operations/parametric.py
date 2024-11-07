from __future__ import annotations

from logging import getLogger

import numpy as np
import sympy
import torch
from torch import Tensor, cdouble

from qadence.blocks import (
    AbstractBlock,
    ParametricBlock,
)
from qadence.blocks.utils import (
    add,  # noqa
    chain,
)
from qadence.noise import NoiseHandler
from qadence.parameters import (
    Parameter,
    ParamMap,
    evaluate,
)
from qadence.types import OpName, TNumber, TParameter

from .primitive import I, X, Y, Z

logger = getLogger(__name__)


class PHASE(ParametricBlock):
    """The Parametric Phase / S gate."""

    name = OpName.PHASE

    def __init__(
        self,
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.parameters = ParamMap(parameter=parameter)
        self.generator = I(target) - Z(target)
        super().__init__((target,), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([0, 2], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        lmbda = torch.exp(1j * evaluate(self.parameters.parameter, as_torch=True))
        return torch.cat((torch.ones(1), lmbda))


class RX(ParametricBlock):
    """The Rx gate."""

    name = OpName.RX

    def __init__(
        self,
        target: int,
        parameter: Parameter | TParameter | ParamMap,
        noise: NoiseHandler | None = None,
    ) -> None:
        # TODO: should we give them more meaningful names? like 'angle'?
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = X(target)
        super().__init__((target,), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((lmbd, lmbd.conj()))


class RY(ParametricBlock):
    """The Ry gate."""

    name = OpName.RY

    def __init__(
        self,
        target: int,
        parameter: Parameter | TParameter | ParamMap,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = Y(target)
        super().__init__((target,), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((lmbd, lmbd.conj()))


class RZ(ParametricBlock):
    """The Rz gate."""

    name = OpName.RZ

    def __init__(
        self,
        target: int,
        parameter: Parameter | TParameter | ParamMap,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = Z(target)
        super().__init__((target,), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 1

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        val = evaluate(self.parameters.parameter, as_torch=True)
        lmbd = torch.cos(val / 2.0) - 1j * torch.sin(val / 2.0)
        return torch.cat((lmbd, lmbd.conj()))


class U(ParametricBlock):
    """Arbitrary one-qubit rotation in the Bloch sphere.

    This operation accepts 3 parameters (phi, theta, omega)
    """

    name = OpName.U

    def __init__(
        self,
        target: int,
        phi: Parameter | TParameter,
        theta: Parameter | TParameter,
        omega: Parameter | TParameter,
        noise: NoiseHandler | None = None,
    ) -> None:
        self.parameters = ParamMap(phi=phi, theta=theta, omega=omega)
        self.generator = chain(Z(target), Y(target), Z(target))
        super().__init__((target,), noise=noise)

    @classmethod
    def num_parameters(cls) -> int:
        return 3

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=torch.cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        phi = evaluate(self.parameters.phi)
        theta = evaluate(self.parameters.theta)
        omega = evaluate(self.parameters.omega)
        lmbd = np.exp(-1j * (phi + omega) / 2) * np.cos(theta / 2)
        return torch.cat((lmbd, lmbd.conj()))

    @property
    def n_qubits(self) -> int:
        return 1

    def digital_decomposition(self) -> AbstractBlock:
        return chain(
            RZ(self.qubit_support[0], self.parameters.phi),
            RY(self.qubit_support[0], self.parameters.theta),
            RZ(self.qubit_support[0], self.parameters.omega),
        )
