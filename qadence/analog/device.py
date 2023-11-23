from __future__ import annotations

from dataclasses import dataclass, fields

from torch import pi

from qadence.types import DeviceType, Interaction


@dataclass(frozen=True, eq=True)
class RydbergDevice:
    """
    Dataclass for interacting Rydberg atoms under an Hamiltonian:

    H = ∑ᵢ(Ω/2 cos(φ)*Xᵢ - sin(φ)*Yᵢ - δnᵢ) + H_int,

    where:

    H_int = ∑_(j<i) (C_6 / R**6) * kron(N_i, N_j) for the NN interaction;

    H_int = ∑_(j<i) (C_3 / R**3) * (kron(X_i, X_j) + kron(Y_i, Y_j)) for the XY interaction;
    """

    interaction: Interaction = Interaction.NN
    """Defines the interaction Hamiltonian."""

    rydberg_level: int = 60
    """Rydberg level affecting the value of C_6."""

    coeff_xy: float = 3700.00
    """Value of C_3."""

    max_abs_detuning: float = 2 * pi * 4
    """Maximum value of the detuning δ."""

    max_amp: float = 2 * pi * 3
    """Maximum value of the amplitude Ω."""

    pattern: None = None
    """Addressing pattern to be added."""

    device_type: DeviceType = DeviceType.IDEALIZED
    """DeviceType.IDEALIZED or REALISTIC to convert to the Pulser backend."""

    def __post_init__(self) -> None:
        # FIXME: Currently not supporting custom interaction functions.
        if self.interaction not in [Interaction.NN, Interaction.XY]:
            raise KeyError(
                "RydbergDevice currently only supports Interaction.NN or Interaction.XY."
            )

    def _to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def _from_dict(cls, d: dict) -> RydbergDevice:
        return cls(**d)


def IdealDevice(pattern: None = None) -> RydbergDevice:
    return RydbergDevice(
        interaction=Interaction.NN,
        rydberg_level=60,
        coeff_xy=3700.00,
        max_abs_detuning=2 * pi * 4,
        max_amp=2 * pi * 3,
        pattern=pattern,
        device_type=DeviceType.IDEALIZED,
    )


def RealisticDevice(pattern: None = None) -> RydbergDevice:
    return RydbergDevice(
        interaction=Interaction.NN,
        rydberg_level=60,
        coeff_xy=3700.00,
        max_abs_detuning=2 * pi * 4,
        max_amp=2 * pi * 3,
        pattern=pattern,
        device_type=DeviceType.REALISTIC,
    )
