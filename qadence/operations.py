"""Basic operations to be implemented by backends."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Tuple, Union

import numpy as np
import sympy
import torch
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.tree import Tree
from sympy import Basic
from torch import Tensor, cdouble, tensor

from qadence.blocks import (
    AbstractBlock,
    ControlBlock,
    ParametricBlock,
    ParametricControlBlock,
    PrimitiveBlock,
    TimeEvolutionBlock,
)
from qadence.blocks.analog import (
    AnalogBlock,
    ConstantAnalogRotation,
    QubitSupport,
    WaitBlock,
)
from qadence.blocks.block_to_tensor import block_to_tensor
from qadence.blocks.primitive import ProjectorBlock
from qadence.blocks.utils import (
    add,  # noqa
    block_is_commuting_hamiltonian,
    block_is_qubit_hamiltonian,
    chain,
    expressions,
    kron,
)
from qadence.decompose import lie_trotter_suzuki
from qadence.logger import get_logger
from qadence.parameters import (
    Parameter,
    ParamMap,
    evaluate,
    extract_original_param_entry,
)
from qadence.types import LTSOrder, OpName, TGenerator, TNumber, TParameter
from qadence.utils import eigenvalues

logger = get_logger(__name__)


# Modules to be automatically added to the qadence namespace
__all__ = [
    "X",
    "Y",
    "Z",
    "N",
    "H",
    "I",
    "Zero",
    "RX",
    "RY",
    "RZ",
    "U",
    "CNOT",
    "CZ",
    "MCZ",
    "HamEvo",
    "CRX",
    "MCRX",
    "CRY",
    "MCRY",
    "CRZ",
    "MCRZ",
    "T",
    "TDagger",
    "S",
    "SDagger",
    "SWAP",
    "PHASE",
    "CPHASE",
    "CSWAP",
    "MCPHASE",
    "Toffoli",
    "wait",
    "entangle",
    "AnalogEntanglement",
    "AnalogRot",
    "AnalogRX",
    "AnalogRY",
    "AnalogRZ",
    "AnalogSWAP",
]


class X(PrimitiveBlock):
    """The X gate."""

    name = OpName.X

    def __init__(self, target: int):
        super().__init__((target,))

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class Y(PrimitiveBlock):
    """The Y gate."""

    name = OpName.Y

    def __init__(self, target: int):
        super().__init__((target,))

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class Z(PrimitiveBlock):
    """The Z gate."""

    name = OpName.Z

    def __init__(self, target: int):
        super().__init__((target,))

    @property
    def generator(self) -> AbstractBlock:
        return self

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([-1, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([-1, 1], dtype=cdouble)


class Projector(ProjectorBlock):
    """The projector operator."""

    name = OpName.PROJ

    def __init__(
        self,
        ket: str,
        bra: str,
        qubit_support: int | tuple[int, ...],
    ):
        super().__init__(ket=ket, bra=bra, qubit_support=qubit_support)

    @property
    def generator(self) -> None:
        raise ValueError("Property `generator` not available for non-unitary operator.")

    @property
    def eigenvalues_generator(self) -> None:
        raise ValueError("Property `eigenvalues_generator` not available for non-unitary operator.")


class N(Projector):
    """The N = (1/2)(I-Z) operator."""

    name = OpName.N

    def __init__(self, target: int, state: str = "1"):
        super().__init__(ket=state, bra=state, qubit_support=(target,))

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

    def __init__(self, target: int):
        self.generator = I(target) - Z(target)
        super().__init__((target,))

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([1, 1j], dtype=cdouble)

    def dagger(self) -> SDagger:
        return SDagger(*self.qubit_support)


class SDagger(PrimitiveBlock):
    """The Hermitian adjoint/conjugate transpose of the S / Phase gate."""

    name = OpName.SDAGGER

    def __init__(self, target: int):
        self.generator = I(target) - Z(target)
        super().__init__((target,))

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor([0, 1], dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return tensor([1, -1j], dtype=cdouble)

    def dagger(self) -> S:
        return S(*self.qubit_support)


class PHASE(ParametricBlock):
    """The Parametric Phase / S gate."""

    name = OpName.PHASE

    def __init__(self, target: int, parameter: Parameter | TNumber | sympy.Expr | str):
        self.parameters = ParamMap(parameter=parameter)
        self.generator = I(target) - Z(target)
        super().__init__((target,))

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


class I(PrimitiveBlock):
    """The identity gate."""

    name = OpName.I

    def __init__(self, target: int):
        super().__init__((target,))

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
        return I(*self.qubit_support)

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.ones(2, dtype=cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.ones(2, dtype=cdouble)

    def __ascii__(self, console: Console) -> Padding:
        return Padding("──────", (1, 1, 1, 1))


TPauliBlock = Union[X, Y, Z, I, N]


class H(PrimitiveBlock):
    """The Hadamard or H gate."""

    name = OpName.H

    def __init__(self, target: int):
        self.generator = (1 / np.sqrt(2)) * (X(target) + Z(target) - np.sqrt(2) * I(target))
        super().__init__((target,))

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


class RX(ParametricBlock):
    """The Rx gate."""

    name = OpName.RX

    def __init__(self, target: int, parameter: Parameter | TParameter | ParamMap):
        # TODO: should we give them more meaningful names? like 'angle'?
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = X(target)
        super().__init__((target,))

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

    def __init__(self, target: int, parameter: Parameter | TParameter | ParamMap):
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = Y(target)
        super().__init__((target,))

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

    def __init__(self, target: int, parameter: Parameter | TParameter | ParamMap):
        self.parameters = (
            parameter if isinstance(parameter, ParamMap) else ParamMap(parameter=parameter)
        )
        self.generator = Z(target)
        super().__init__((target,))

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
    ):
        self.parameters = ParamMap(phi=phi, theta=theta, omega=omega)
        self.generator = chain(Z(target), Y(target), Z(target))
        super().__init__((target,))

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


class HamEvo(TimeEvolutionBlock):
    """
    A block implementing the Hamiltonian evolution operation H where:

        H = exp(-iG, t)
    where G represents a square generator and t represents the time parameter
    which can be parametrized.

    Arguments:
        generator: Either a AbstractBlock, torch.Tensor or numpy.ndarray.
        parameter: A scalar or vector of numeric or torch.Tensor type.
        qubit_support: The qubits on which the evolution will be performed on.

    Examples:

    ```python exec="on" source="material-block" result="json"
    from qadence import RX, HamEvo, run
    import torch
    hevo = HamEvo(generator=RX(0, torch.pi), parameter=torch.rand(2))
    print(run(hevo))
    # Now lets use a torch.Tensor as a generator, Now we have to pass the support
    gen = torch.rand(2,2, dtype=torch.complex128)
    hevo = HamEvo(generator=gen, parameter=torch.rand(2), qubit_support=(0,))
    print(run(hevo))
    ```
    """

    name = OpName.HAMEVO
    draw_generator: bool = False

    def __init__(
        self,
        generator: Union[TGenerator, AbstractBlock],
        parameter: TParameter,
        qubit_support: tuple[int, ...] = None,
    ):
        gen_exprs = {}
        if qubit_support is None and not isinstance(generator, AbstractBlock):
            raise ValueError("You have to supply a qubit support for non-block generators.")
        super().__init__(qubit_support if qubit_support else generator.qubit_support)
        if isinstance(generator, AbstractBlock):
            qubit_support = generator.qubit_support
            if generator.is_parametric:
                gen_exprs = {str(e): e for e in expressions(generator)}
        elif isinstance(generator, torch.Tensor):
            msg = "Please provide a square generator."
            if len(generator.shape) == 2:
                assert generator.shape[0] == generator.shape[1], msg
            elif len(generator.shape) == 3:
                assert generator.shape[1] == generator.shape[2], msg
                assert generator.shape[0] == 1, "Qadence doesnt support batched generators."
            else:
                raise TypeError(
                    "Only 2D or 3D generators are supported.\
                                In case of a 3D generator, the batch dim\
                                is expected to be at dim 0."
                )
            gen_exprs = {str(generator.__hash__()): generator}
        elif isinstance(generator, (sympy.Basic, sympy.Array)):
            gen_exprs = {str(generator): generator}
        else:
            raise TypeError(
                f"Generator of type {type(generator)} not supported.\
                            If you're using a numpy.ndarray, please cast it to a torch tensor."
            )
        ps = {"parameter": Parameter(parameter), **gen_exprs}
        self.parameters = ParamMap(**ps)
        self.generator = generator

    @classmethod
    def num_parameters(cls) -> int:
        return 2

    @cached_property
    def eigenvalues_generator(
        self, max_num_evals: int | None = None, max_num_gaps: int | None = None
    ) -> Tensor:
        if isinstance(self.generator, AbstractBlock):
            generator_tensor = block_to_tensor(self.generator)
        elif isinstance(self.generator, Tensor):
            generator_tensor = self.generator
        return eigenvalues(generator_tensor, max_num_evals, max_num_gaps)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.exp(
            -1j * evaluate(self.parameters.parameter, as_torch=True) * self.eigenvalues_generator
        )

    @property
    def n_qubits(self) -> int:
        if isinstance(self.generator, Tensor):
            n_qubits = int(np.log2(self.generator.shape[-1]))
        else:
            n_qubits = self.generator.n_qubits  # type: ignore [union-attr]

        return n_qubits

    def dagger(self) -> Any:
        p = list(self.parameters.expressions())[0]
        return HamEvo(deepcopy(self.generator), -extract_original_param_entry(p))

    def digital_decomposition(self, approximation: LTSOrder = LTSOrder.ST4) -> AbstractBlock:
        """Decompose the Hamiltonian evolution into digital gates.

        Args:
            approximation (str, optional): Choose the type of decomposition. Defaults to "st4".
                Available types are:
                * 'basic' = apply first-order Trotter formula and decompose each term of
                    the exponential into digital gates. It is exact only if applied to an
                    operator whose terms are mutually commuting.
                * 'st2' = Trotter-Suzuki 2nd order formula for approximating non-commuting
                    Hamiltonians.
                * 'st4' = Trotter-Suzuki 4th order formula for approximating non-commuting
                    Hamiltonians.

        Returns:
            AbstractBlock: a block with the digital decomposition
        """

        # psi(t) = exp(-i * H * t * psi0)
        # psi(t) = exp(-i * lambda * t * psi0)
        # H = sum(Paulin) + sum(Pauli1*Pauli2)
        logger.info("Quantum simulation of the time-independent Schrödinger equation.")

        blocks = []

        # how to change the type/dict to enum effectively

        # when there is a term including non-commuting matrices use st2 or st4

        # 1) should check that the given generator respects the constraints
        # single-qubit gates

        assert isinstance(
            self.generator, AbstractBlock
        ), "Only a generator represented as a block can be decomposed"

        if block_is_qubit_hamiltonian(self.generator):
            try:
                block_is_commuting_hamiltonian(self.generator)
                approximation = LTSOrder.BASIC  # use the simpler approach if the H is commuting
            except TypeError:
                logger.warning(
                    """Non-commuting terms in the Pauli operator.
                    The Suzuki-Trotter approximation is applied."""
                )

            blocks.extend(
                lie_trotter_suzuki(
                    block=self.generator,
                    parameter=self.parameters.parameter,
                    order=LTSOrder[approximation],
                )
            )

            # 2) return an AbstractBlock instance with the set of gates
            # resulting from the decomposition

            return chain(*blocks)
        else:
            raise NotImplementedError(
                "The current digital decomposition can be applied only to Pauli Hamiltonians."
            )


class CNOT(ControlBlock):
    """The CNot, or CX, gate."""

    name = OpName.CNOT

    def __init__(self, control: int, target: int) -> None:
        self.generator = kron(N(control), X(target) - I(target))
        super().__init__((control,), X(target))

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

    def __init__(self, control: tuple[int, ...], target: int) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target) - I(target))
        super().__init__(control, Z(target))

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

    def __init__(self, control: int, target: int) -> None:
        super().__init__((control,), target)


class MCRX(ParametricControlBlock):
    name = OpName.MCRX

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], X(target))
        super().__init__(control, RX(target, parameter))

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
    ):
        super().__init__((control,), target, parameter)


class MCRY(ParametricControlBlock):
    name = OpName.MCRY

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Y(target))
        super().__init__(control, RY(target, parameter))

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
        self,
        control: int,
        target: int,
        parameter: TParameter,
    ):
        super().__init__((control,), target, parameter)


class MCRZ(ParametricControlBlock):
    name = OpName.MCRZ

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target))
        super().__init__(control, RZ(target, parameter))

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
    ):
        super().__init__((control,), target, parameter)


class CSWAP(ControlBlock):
    """The CSWAP (Control-SWAP) gate."""

    name = OpName.CSWAP

    def __init__(self, control: int | tuple[int, ...], target1: int, target2: int) -> None:
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
        super().__init__((control,), SWAP(target1, target2))

    @property
    def eigenvalues_generator(self) -> Tensor:
        return torch.tensor((1, -1, 1, 1, 1, 1, 1, 1), dtype=torch.cdouble)

    @property
    def eigenvalues(self) -> Tensor:
        return torch.tensor((1, -1, 1, 1, 1, 1, 1, 1), dtype=torch.cdouble)

    @property
    def nqubits(self) -> int:
        return 3


class T(PrimitiveBlock):
    """The T gate."""

    name = OpName.T

    def __init__(self, target: int):
        self.generator = I(target) - Z(target)
        super().__init__((target,))

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
        return TDagger(*self.qubit_support)


class TDagger(PrimitiveBlock):
    """The Hermitian adjoint/conjugate transpose of the T gate."""

    # FIXME: this gate is not support by any backend
    name = "T_dagger"

    def __init__(self, target: int):
        self.generator = I(target) - Z(target)
        super().__init__((target,))

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
        return T(*self.qubit_support)


class SWAP(PrimitiveBlock):
    """The SWAP gate."""

    name = OpName.SWAP

    def __init__(self, control: int, target: int) -> None:
        a11 = 0.5 * (Z(control) - I(control))
        a22 = -0.5 * (Z(target) + I(target))
        a12 = 0.5 * (chain(X(control), Z(control)) + X(control))
        a21 = 0.5 * (chain(Z(target), X(target)) + X(target))
        self.generator = (
            kron(-1.0 * a22, a11) + kron(-1.0 * a11, a22) + kron(a12, a21) + kron(a21, a12)
        )
        super().__init__((control, target))

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


class AnalogSWAP(HamEvo):
    """
    Single time-independent Hamiltonian evolution over a Rydberg Ising.

    hamiltonian yielding a SWAP (up to global phase).

    Derived from
    [Bapat et al.](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L012023)
    where it is applied to XX-type Hamiltonian
    """

    name = OpName.ANALOGSWAP

    def __init__(self, control: int, target: int, parameter: TParameter = 3 * np.pi / 4):
        rydberg_ising_hamiltonian_generator = (
            4.0 * kron((I(control) - Z(control)) / 2.0, (I(target) - Z(target)) / 2.0)
            + (2.0 / 3.0) * np.sqrt(2.0) * X(control)
            + (2.0 / 3.0) * np.sqrt(2.0) * X(target)
            + (1.0 + np.sqrt(5.0) / 3) * Z(control)
            + (1.0 + np.sqrt(5.0) / 3) * Z(target)
        )
        super().__init__(rydberg_ising_hamiltonian_generator, parameter, (control, target))


class MCPHASE(ParametricControlBlock):
    name = OpName.MCPHASE

    def __init__(
        self,
        control: tuple[int, ...],
        target: int,
        parameter: Parameter | TNumber | sympy.Expr | str,
    ) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], Z(target) - I(target))
        super().__init__(control, PHASE(target, parameter))

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
    ):
        super().__init__((control,), target, parameter)


class Toffoli(ControlBlock):
    name = OpName.TOFFOLI

    def __init__(self, control: tuple[int, ...], target: int) -> None:
        self.generator = kron(*[N(qubit) for qubit in control], X(target) - I(target))
        super().__init__(control, X(target))

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


# FIXME: better name that stresses difference to `Wait`?
@dataclass(eq=False, repr=False)
class AnalogEntanglement(AnalogBlock):
    parameters: ParamMap = ParamMap(duration=1.0)
    qubit_support: QubitSupport = QubitSupport("global")

    @property
    def eigenvalues_generator(self) -> torch.Tensor:
        return torch.tensor([0.0], dtype=cdouble)

    @property
    def duration(self) -> Basic:
        return self.parameters.duration


def _cast(T: Any, val: Any) -> Any:
    return val if isinstance(val, T) else T(val)


def wait(
    duration: TNumber | sympy.Basic,
    qubit_support: str | QubitSupport | tuple = "global",
    add_pattern: bool = True,
) -> WaitBlock:
    """Constructs a [`WaitBlock`][qadence.blocks.analog.WaitBlock].

    Arguments:
        duration: Time to wait in nanoseconds.
        qubit_support: Qubits the `WaitBlock` is applied to. Can be either
            `"global"` to apply the wait block to all qubits or a tuple of integers.

    Returns:
        a `WaitBlock`
    """
    q = _cast(QubitSupport, qubit_support)
    ps = ParamMap(duration=duration)
    return WaitBlock(parameters=ps, qubit_support=q, add_pattern=add_pattern)


def entangle(
    duration: Any,
    qubit_support: str | QubitSupport | Tuple = "global",
) -> AnalogEntanglement:
    q = _cast(QubitSupport, qubit_support)
    ps = ParamMap(duration=duration)
    return AnalogEntanglement(parameters=ps, qubit_support=q)


def AnalogRot(
    duration: float | str | Parameter = 1000.0,
    omega: float | str | Parameter = 0,
    delta: float | str | Parameter = 0,
    phase: float | str | Parameter = 0,
    qubit_support: str | QubitSupport | Tuple = "global",
    add_pattern: bool = True,
) -> ConstantAnalogRotation:
    """General analog rotation operation.

    Arguments:
        duration: Duration of the rotation [ns].
        omega: Rotation frequency [rad/μs]
        delta: Rotation frequency [rad/μs]
        phase: Phase angle [rad]
        qubit_support: Defines the (local/global) qubit support

    Returns:
        ConstantAnalogRotation
    """
    q = _cast(QubitSupport, qubit_support)
    duration = Parameter(duration)
    omega = Parameter(omega)
    delta = Parameter(delta)
    phase = Parameter(phase)
    alpha = duration * sympy.sqrt(omega**2 + delta**2) / 1000
    ps = ParamMap(alpha=alpha, duration=duration, omega=omega, delta=delta, phase=phase)
    return ConstantAnalogRotation(parameters=ps, qubit_support=q, add_pattern=add_pattern)


def _analog_rot(
    angle: float | str | Parameter,
    qubit_support: str | QubitSupport | Tuple,
    phase: float,
    add_pattern: bool = True,
) -> ConstantAnalogRotation:
    q = _cast(QubitSupport, qubit_support)
    # assuming some arbitrary omega = π rad/μs
    alpha = _cast(Parameter, angle)

    omega = np.pi
    duration = alpha / omega * 1000

    # FIXME: once https://github.com/pasqal-io/qadence/issues/150 is fixed set default duration
    # in the function arguments to:
    # duration = Parameter(160)
    # and compute omega like this:
    # omega = alpha / duration * 1000
    ps = ParamMap(alpha=alpha, duration=duration, omega=omega, delta=0, phase=phase)
    return ConstantAnalogRotation(parameters=ps, qubit_support=q, add_pattern=add_pattern)


def AnalogRX(
    angle: float | str | Parameter,
    qubit_support: str | QubitSupport | Tuple = "global",
    add_pattern: bool = True,
) -> ConstantAnalogRotation:
    """Analog X rotation.

    Shorthand for [`AnalogRot`][qadence.operations.AnalogRot]:

    ```python
    φ=2.4; Ω=π; t = φ/Ω * 1000
    AnalogRot(duration=t, omega=Ω)
    ```

    Arguments:
        angle: Rotation angle [rad]
        qubit_support: Defines the (local/global) qubit support

    Returns:
        ConstantAnalogRotation
    """
    return _analog_rot(angle, qubit_support, phase=0, add_pattern=add_pattern)


def AnalogRY(
    angle: float | str | Parameter,
    qubit_support: str | QubitSupport | Tuple = "global",
    add_pattern: bool = True,
) -> ConstantAnalogRotation:
    """Analog Y rotation.

    Shorthand for [`AnalogRot`][qadence.operations.AnalogRot]:

    ```python
    φ=2.4; Ω=π; t = φ/Ω * 1000
    AnalogRot(duration=t, omega=Ω, phase=-π/2)
    ```
    Arguments:
        angle: Rotation angle [rad]
        qubit_support: Defines the (local/global) qubit support

    Returns:
        ConstantAnalogRotation
    """
    return _analog_rot(angle, qubit_support, phase=-np.pi / 2, add_pattern=add_pattern)


def AnalogRZ(
    angle: float | str | Parameter,
    qubit_support: str | QubitSupport | Tuple = "global",
    add_pattern: bool = True,
) -> ConstantAnalogRotation:
    """Analog Z rotation. Shorthand for [`AnalogRot`][qadence.operations.AnalogRot]:
    ```
    φ=2.4; δ=π; t = φ/δ * 100)
    AnalogRot(duration=t, delta=δ, phase=π/2)
    ```
    """
    q = _cast(QubitSupport, qubit_support)
    alpha = _cast(Parameter, angle)
    delta = np.pi
    duration = alpha / delta * 1000
    ps = ParamMap(alpha=alpha, duration=duration, omega=0, delta=delta, phase=0.0)
    return ConstantAnalogRotation(qubit_support=q, parameters=ps, add_pattern=add_pattern)


# gate sets
# FIXME: this could be inferred by the number of qubits if we had
# a class property for each operation. The number of qubits can default
# to None for operations which do not have it by default
# this would allow to greatly simplify the tests
pauli_gateset: list = [I, X, Y, Z]
# FIXME: add Tdagger when implemented
single_qubit_gateset = [X, Y, Z, H, I, RX, RY, RZ, U, S, SDagger, T, PHASE]
two_qubit_gateset = [CNOT, SWAP, CZ, CRX, CRY, CRZ, CPHASE]
three_qubit_gateset = [CSWAP]
multi_qubit_gateset = [Toffoli, MCRX, MCRY, MCRZ, MCPHASE, MCZ]
analog_gateset = [
    HamEvo,
    ConstantAnalogRotation,
    AnalogEntanglement,
    AnalogSWAP,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    entangle,
    wait,
]
non_unitary_gateset = [Zero, N, Projector]
