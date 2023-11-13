from __future__ import annotations

from typing import Callable, Optional

from dataclasses import dataclass

from qadence.register import Register
from qadence.types import Interaction


@dataclass
class QubitDevice:

    register: Register

    interaction: Interaction

    spacing: float = 1.0

    def __post_init__(self) -> None:
        if self.spacing != 1.0:
            self.register = self.register._scale_positions(self.spacing)


@dataclass
class RydbergDevice(QubitDevice):

    interaction: Interaction = Interaction.NN

    rydberg_level: int = 60

    coeff_xy: float = 3700.00

    def __post_init__(self) -> None:
        # FIXME: Currently not supporting custom interaction functions.
        if self.interaction not in [Interaction.NN, Interaction.XY]:
            raise KeyError(
                "RydbergDevice currently only supports Interaction.NN or Interaction.XY."
            )
        
        super().__post_init__()