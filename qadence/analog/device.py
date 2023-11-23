from __future__ import annotations

from dataclasses import dataclass, fields

from torch import pi

from qadence.types import DeviceType, Interaction


@dataclass(frozen=True, eq=True)
class RydbergDevice:
    """Dataclass for interacting Rydberg atoms."""

    interaction: Interaction = Interaction.NN

    rydberg_level: int = 60

    coeff_xy: float = 3700.00

    max_abs_detuning: float = 2 * pi * 4

    max_amp: float = 2 * pi * 3

    pattern: None = None

    device_type: DeviceType = DeviceType.IDEALIZED

    def __post_init__(self) -> None:
        # FIXME: Currently not supporting custom interaction functions.
        if self.interaction not in [Interaction.NN, Interaction.XY]:
            raise KeyError(
                "RydbergDevice currently only supports Interaction.NN or Interaction.XY."
            )

    def _to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def _from_dict(cls, d: dict) -> RydbergDevice | None:
        return cls(**d) if d else None


def IdealDevice(pattern: None = None) -> RydbergDevice:
    return RydbergDevice(
        interaction=Interaction.NN,
        rydberg_level=60,
        max_abs_detuning=2 * pi * 4,
        max_amp=2 * pi * 3,
        pattern=pattern,
        device_type=DeviceType.IDEALIZED,
    )


def RealisticDevice(pattern: None = None) -> RydbergDevice:
    return RydbergDevice(
        interaction=Interaction.NN,
        rydberg_level=60,
        max_abs_detuning=2 * pi * 4,
        max_amp=2 * pi * 3,
        pattern=pattern,
        device_type=DeviceType.REALISTIC,
    )
