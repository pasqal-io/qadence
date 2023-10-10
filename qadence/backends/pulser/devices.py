from __future__ import annotations

from numpy import pi
from pulser.channels.channels import Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices._device_datacls import Device as PulserDevice
from pulser.devices._device_datacls import VirtualDevice

from qadence.types import StrEnum

# Idealized virtual device
IdealDevice = VirtualDevice(
    name="IdealizedDevice",
    dimensions=2,
    rydberg_level=60,
    max_atom_num=100,
    max_radial_distance=100,
    min_atom_distance=0,
    channel_objects=(
        Rydberg.Global(max_abs_detuning=2 * pi * 4, max_amp=2 * pi * 3),
        Rydberg.Local(max_targets=1000, max_abs_detuning=2 * pi * 4, max_amp=2 * pi * 3),
    ),
)


# device with realistic specs with local channels and custom bandwith.
RealisticDevice = PulserDevice(
    name="RealisticDevice",
    dimensions=2,
    rydberg_level=60,
    max_atom_num=100,
    max_radial_distance=60,
    min_atom_distance=5,
    channel_objects=(
        Rydberg.Global(
            max_abs_detuning=2 * pi * 4,
            max_amp=2 * pi * 1.5,
            clock_period=4,
            min_duration=16,
            max_duration=4000,
            mod_bandwidth=16,
            eom_config=RydbergEOM(
                limiting_beam=RydbergBeam.RED,
                max_limiting_amp=40 * 2 * pi,
                intermediate_detuning=700 * 2 * pi,
                mod_bandwidth=24,
                controlled_beams=(RydbergBeam.BLUE,),
            ),
        ),
        Rydberg.Local(
            max_targets=20,
            max_abs_detuning=2 * pi * 4,
            max_amp=2 * pi * 3,
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
            mod_bandwidth=16,
            eom_config=RydbergEOM(
                limiting_beam=RydbergBeam.RED,
                max_limiting_amp=40 * 2 * pi,
                intermediate_detuning=700 * 2 * pi,
                mod_bandwidth=24,
                controlled_beams=(RydbergBeam.BLUE,),
            ),
        ),
    ),
)


class Device(StrEnum):
    """Supported types of devices for Pulser backend"""

    IDEALIZED = IdealDevice
    "idealized device, least realistic"

    REALISTIC = RealisticDevice
    "device with realistic specs"
