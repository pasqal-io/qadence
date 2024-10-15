from __future__ import annotations

from qadence import DigitalNoise
from qadence.noise.protocols import digital_noise_protocols
from qadence.serialization import deserialize


def test_serialization_noise() -> None:
    for noise in digital_noise_protocols:
        op_noise = DigitalNoise(noise, error_probability=0.1)
        dnoise = op_noise._to_dict()
        op_from_dnoise = deserialize(dnoise)
        assert op_from_dnoise == op_noise
