from __future__ import annotations

from dataclasses import dataclass

from torch import pi

from qadence.types import Interaction


@dataclass
class RydbergDevice:
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
