from __future__ import annotations

from pulser.channels.channels import Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices._device_datacls import VirtualDevice

from qadence.types import PI


# Idealized virtual device
def IdealDevice(
    rydberg_level: int = 60, max_abs_detuning: float = 2 * PI * 4, max_amp: float = 2 * PI * 3
) -> VirtualDevice:
    return VirtualDevice(
        name="IdealizedDevice",
        dimensions=2,
        rydberg_level=rydberg_level,
        max_atom_num=100,
        max_radial_distance=100,
        min_atom_distance=0,
        channel_objects=(
            Rydberg.Global(max_abs_detuning=max_abs_detuning, max_amp=max_amp),
            Rydberg.Local(max_targets=1000, max_abs_detuning=max_abs_detuning, max_amp=max_amp),
        ),
    )


# Device with realistic specs with local channels and custom bandwith.
def RealisticDevice(
    rydberg_level: int = 60, max_abs_detuning: float = 2 * PI * 4, max_amp: float = 2 * PI * 3
) -> VirtualDevice:
    return VirtualDevice(
        name="RealisticDevice",
        dimensions=2,
        rydberg_level=rydberg_level,
        max_atom_num=100,
        max_radial_distance=60,
        min_atom_distance=5,
        channel_objects=(
            Rydberg.Global(
                max_abs_detuning=max_abs_detuning,
                max_amp=max_amp * 0.5,
                clock_period=4,
                min_duration=16,
                max_duration=4000,
                mod_bandwidth=16,
                eom_config=RydbergEOM(
                    limiting_beam=RydbergBeam.RED,
                    max_limiting_amp=40 * 2 * PI,
                    intermediate_detuning=700 * 2 * PI,
                    mod_bandwidth=24,
                    controlled_beams=(RydbergBeam.BLUE,),
                ),
            ),
            Rydberg.Local(
                max_targets=20,
                max_abs_detuning=max_abs_detuning,
                max_amp=max_amp,
                clock_period=4,
                min_duration=16,
                max_duration=2**26,
                mod_bandwidth=16,
                eom_config=RydbergEOM(
                    limiting_beam=RydbergBeam.RED,
                    max_limiting_amp=40 * 2 * PI,
                    intermediate_detuning=700 * 2 * PI,
                    mod_bandwidth=24,
                    controlled_beams=(RydbergBeam.BLUE,),
                ),
            ),
        ),
    )
