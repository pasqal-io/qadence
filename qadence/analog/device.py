from __future__ import annotations

from dataclasses import dataclass

from torch import pi

from qadence.register import Register
from qadence.types import Interaction


@dataclass
class QubitDevice:
    """
    General dataclass for defining devices of interacting qubits.

    Subclass it to define device-specific constants, methods and data-validations.
    """

    register: Register

    interaction: Interaction

    spacing: float = 1.0

    def __post_init__(self) -> None:
        if self.spacing != 1.0:
            self.register = self.register._scale_positions(self.spacing)


@dataclass
class RydbergDevice(QubitDevice):
    """Dataclass for interacting Rydberg atoms."""

    interaction: Interaction = Interaction.NN

    rydberg_level: int = 60

    coeff_xy: float = 3700.00

    max_abs_detuning: float = 2 * pi * 4

    max_amp: float = 2 * pi * 3

    def __post_init__(self) -> None:
        # FIXME: Currently not supporting custom interaction functions.
        if self.interaction not in [Interaction.NN, Interaction.XY]:
            raise KeyError(
                "RydbergDevice currently only supports Interaction.NN or Interaction.XY."
            )

        super().__post_init__()
