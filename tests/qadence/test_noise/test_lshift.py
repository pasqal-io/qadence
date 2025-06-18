from __future__ import annotations

from qadence import RX, X, Y
from qadence.noise import available_protocols

def test_equality() -> None:
    noise = available_protocols.Bitflip(error_definition=0.1)
    rx = RX(0, 'theta')
    rx_noise = RX(0, 'theta', noise=noise)
    assert rx != rx_noise

    rx = rx << noise
    assert rx == rx_noise

    combined_noise = noise | available_protocols.Bitflip(error_definition=0.2)
    rx << combined_noise
    # this resets the noise
    assert rx != rx_noise
    assert rx.noise == combined_noise
