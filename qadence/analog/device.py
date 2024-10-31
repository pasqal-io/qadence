from __future__ import annotations

from dataclasses import dataclass, fields

from qadence.analog import AddressingPattern
from qadence.types import PI, DeviceType, Interaction


@dataclass(frozen=True, eq=True)
class RydbergDevice:
    """
    Dataclass for interacting Rydberg atoms under an Hamiltonian:

    H = ∑_i [Ω/2 * (cos(φ) * Xᵢ - sin(φ) * Yᵢ) - δ * N_i] + H_int,

    where:

    H_int = ∑_(j<i) (C_6 / R**6) * (N_i @ N_j) for the NN interaction;

    H_int = ∑_(j<i) (C_3 / R**3) * ((X_i @ X_j) + (Y_i @ Y_j)) for the XY interaction;
    """

    interaction: Interaction = Interaction.NN
    """Defines the interaction Hamiltonian."""

    rydberg_level: int = 60
    """Rydberg level affecting the value of C_6."""

    coeff_xy: float = 3700.00
    """Value of C_3."""

    max_detuning: float = 2 * PI * 4
    """Maximum value of the detuning δ."""

    max_amp: float = 2 * PI * 3
    """Maximum value of the amplitude Ω."""

    pattern: AddressingPattern | None = None
    """Semi-local addressing pattern configuration."""

    type: DeviceType = DeviceType.IDEALIZED
    """DeviceType.IDEALIZED or REALISTIC to convert to the Pulser backend."""

    def __post_init__(self) -> None:
        # FIXME: Currently not supporting custom interaction functions.
        if self.interaction not in [Interaction.NN, Interaction.XY]:
            raise KeyError(
                "RydbergDevice currently only supports Interaction.NN or Interaction.XY."
            )

    def _to_dict(self) -> dict:
        device_dict = dict()
        for field in fields(self):
            if field.name != "pattern":
                device_dict[field.name] = getattr(self, field.name)
            else:
                device_dict[field.name] = (
                    self.pattern._to_dict() if self.pattern is not None else {}
                )
        return device_dict

    @classmethod
    def _from_dict(cls, d: dict) -> RydbergDevice:
        pattern = AddressingPattern._from_dict(d["pattern"])
        d["pattern"] = pattern
        return cls(**d)


def IdealDevice(pattern: AddressingPattern | None = None) -> RydbergDevice:
    return RydbergDevice(
        pattern=pattern,
        type=DeviceType.IDEALIZED,
    )


def RealisticDevice(pattern: AddressingPattern | None = None) -> RydbergDevice:
    return RydbergDevice(
        pattern=pattern,
        type=DeviceType.REALISTIC,
    )
