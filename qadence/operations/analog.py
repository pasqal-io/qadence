from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Tuple

import numpy as np
import sympy
import torch
from sympy import Basic
from torch import cdouble

from qadence.blocks.analog import (
    AnalogBlock,
    ConstantAnalogRotation,
    InteractionBlock,
    QubitSupport,
)
from qadence.blocks.utils import (
    add,  # noqa
    kron,
)
from qadence.parameters import (
    Parameter,
    ParamMap,
)
from qadence.types import PI, OpName, TNumber, TParameter

from .ham_evo import HamEvo
from .primitive import I, X, Z

logger = getLogger(__name__)


class AnalogSWAP(HamEvo):
    """
    Single time-independent Hamiltonian evolution over a Rydberg Ising.

    hamiltonian yielding a SWAP (up to global phase).

    Derived from
    [Bapat et al.](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L012023)
    where it is applied to XX-type Hamiltonian
    """

    name = OpName.ANALOGSWAP

    def __init__(self, control: int, target: int, parameter: TParameter = 3 * PI / 4):
        rydberg_ising_hamiltonian_generator = (
            4.0 * kron((I(control) - Z(control)) / 2.0, (I(target) - Z(target)) / 2.0)
            + (2.0 / 3.0) * np.sqrt(2.0) * X(control)
            + (2.0 / 3.0) * np.sqrt(2.0) * X(target)
            + (1.0 + np.sqrt(5.0) / 3) * Z(control)
            + (1.0 + np.sqrt(5.0) / 3) * Z(target)
        )
        super().__init__(rydberg_ising_hamiltonian_generator, parameter, (control, target))


def _cast(T: Any, val: Any) -> Any:
    return val if isinstance(val, T) else T(val)


def AnalogInteraction(
    duration: TNumber | sympy.Basic,
    qubit_support: str | QubitSupport | tuple = "global",
    add_pattern: bool = True,
) -> InteractionBlock:
    """Evolution of the interaction term for a register of qubits.

    Constructs a [`InteractionBlock`][qadence.blocks.analog.InteractionBlock].

    Arguments:
        duration: Time to evolve the interaction for in nanoseconds.
        qubit_support: Qubits the `InteractionBlock` is applied to. Can be either
            `"global"` to evolve the interaction block to all qubits or a tuple of integers.
        add_pattern: False disables the semi-local addressing pattern
            for the execution of this specific block.

    Returns:
        a `InteractionBlock`
    """
    q = _cast(QubitSupport, qubit_support)
    ps = ParamMap(duration=duration)
    return InteractionBlock(parameters=ps, qubit_support=q, add_pattern=add_pattern)


# FIXME: clarify the usage of this gate, rename more formally, and implement in PyQ
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


def entangle(
    duration: Any,
    qubit_support: str | QubitSupport | Tuple = "global",
) -> AnalogEntanglement:
    q = _cast(QubitSupport, qubit_support)
    ps = ParamMap(duration=duration)
    return AnalogEntanglement(parameters=ps, qubit_support=q)


def AnalogRot(
    duration: float | str | Parameter,
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
        add_pattern: False disables the semi-local addressing pattern
            for the execution of this specific block.

    Returns:
        ConstantAnalogRotation
    """

    if omega == 0 and delta == 0:
        raise ValueError("Parameters omega and delta cannot both be 0.")

    q = _cast(QubitSupport, qubit_support)
    duration = Parameter(duration)
    omega = Parameter(omega)
    delta = Parameter(delta)
    phase = Parameter(phase)
    h_norm = sympy.sqrt(omega**2 + delta**2)
    alpha = duration * h_norm / 1000
    ps = ParamMap(
        alpha=alpha, duration=duration, omega=omega, delta=delta, phase=phase, h_norm=h_norm
    )
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
    delta = 0
    omega = PI
    duration = alpha / omega * 1000
    h_norm = sympy.sqrt(omega**2 + delta**2)

    # FIXME: once https://github.com/pasqal-io/qadence/issues/150 is fixed set default duration
    # in the function arguments to:
    # duration = Parameter(160)
    # and compute omega like this:
    # omega = alpha / duration * 1000
    ps = ParamMap(
        alpha=alpha, duration=duration, omega=omega, delta=delta, phase=phase, h_norm=h_norm
    )
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
    return _analog_rot(angle, qubit_support, phase=-PI / 2, add_pattern=add_pattern)


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
    delta = PI
    omega = 0
    duration = alpha / delta * 1000
    h_norm = sympy.sqrt(omega**2 + delta**2)
    ps = ParamMap(
        alpha=alpha, duration=duration, omega=omega, delta=delta, phase=0.0, h_norm=h_norm
    )
    return ConstantAnalogRotation(qubit_support=q, parameters=ps, add_pattern=add_pattern)
