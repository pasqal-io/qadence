from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Counter, cast

from pyqtorch.noise import NoiseProtocol

from qadence.types import DigitalNoiseType, NoiseProtocolType

PROTOCOL_TO_MODULE = {
    "Readout": "qadence.noise.readout",
}

# Temporary solution
DigitalNoise = NoiseProtocol
digital_noise_protocols = set([DigitalNoiseType(noise.value) for noise in DigitalNoiseType])


@dataclass
class Noise:
    """A container class for all noise protocols."""

    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"
    DEPOLARIZING = "Depolarizing"
    DEPHASING = "Dephasing"
    READOUT = "Readout"

    def __init__(self, protocol: str, options: dict = dict(), type: str = "") -> None:
        self.protocol: str = protocol
        self.options: dict = options
        self.type: str = type

        # forcing in certain cases the type of predefined protocols
        if self.type == "":
            if protocol == "Readout":
                self.type = NoiseProtocolType.READOUT
            if protocol == "Dephasing":
                self.type = NoiseProtocolType.ANALOG
                self.protocol = self.protocol.lower()
            if protocol in digital_noise_protocols:
                self.type = NoiseProtocolType.DIGITAL
        else:
            if self.type not in [NoiseProtocolType(t.value) for t in NoiseProtocolType]:
                raise ValueError("Noise type {self.type} is not supported.")
            if self.type == NoiseProtocolType.ANALOG:
                self.protocol = self.protocol.lower()

    def get_noise_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options, "type": self.type}

    @classmethod
    def _from_dict(cls, d: dict) -> Noise | None:
        if d:
            return cls(d["protocol"], **d["options"], type=d.get("type", ""))
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


def apply_noise(noise: Noise, samples: list[Counter]) -> list[Counter]:
    """Apply noise to samples if READOUT else do not affect.

    Args:
        noise (Noise): Noise instance
        samples (list[Counter]): Samples out of circuit.

    Returns:
        list[Counter]: Changed samples by readout.
    """

    if noise.type == NoiseProtocolType.READOUT:
        error_fn = noise.get_noise_fn()
        # Get the number of qubits from the sample keys.
        n_qubits = len(list(samples[0].keys())[0])
        # Get the number of shots from the sample values.
        n_shots = sum(samples[0].values())
        noisy_samples: list = error_fn(
            counters=samples, n_qubits=n_qubits, options=noise.options, n_shots=n_shots
        )
        return noisy_samples

    return samples
