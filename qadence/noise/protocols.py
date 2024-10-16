from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Counter, cast

from pyqtorch.noise import NoiseProtocol

from qadence.types import DigitalNoiseType, NoiseProtocolType

PROTOCOL_TO_MODULE = {
    "readout": "qadence.noise.readout",
}

# Temporary solution
DigitalNoise = NoiseProtocol
digital_noise_protocols = set([DigitalNoiseType(noise.value) for noise in DigitalNoiseType])


@dataclass
class Noise:
    """A container class for all noise protocols."""

    def __init__(self, protocol: str, options: dict = dict(), type: str = "") -> None:
        self.protocol: str = protocol
        self.options: dict = options
        self.type: str = type

        # forcing in certain cases the type of predefined protocols
        # note that depolarizing exists in both DigitalNoise and PulseNoise
        # albeit the pulse one does not have capital letters
        if self.type == "":
            if protocol == "readout":
                self.type = NoiseProtocolType.POSTPROCESSING
            if protocol in ["dephasing", "depolarizing"]:
                self.type = NoiseProtocolType.PULSE
            if protocol in digital_noise_protocols:
                self.type = NoiseProtocolType.BLOCK

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
            return cls(d["protocol"], **d["options"], type=d["type"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


@dataclass
class PulseNoise(Noise):
    """Pulse noise is pulser-compatible noise where the right options.

    are created for a SimConfig object in Pulser.
    """

    DEPHASING = "dephasing"
    DEPOLARIZING = "depolarizing"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        noise_probs = options.get("noise_probs", None)
        if noise_probs is None:
            raise KeyError("A `noise_probs` option should be passed in options.")
        if not (isinstance(noise_probs, float) or isinstance(noise_probs, Iterable)):
            raise KeyError(
                "A single or a range of noise probabilities"
                " should be passed. Got {type(noise_probs)}."
            )

        super().__init__(protocol, options, NoiseProtocolType.PULSE)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> PulseNoise:
        return cls(d["protocol"], **d["options"])


@dataclass
class PostProcessingNoise(Noise):
    """PostProcessingNoise alters the returned output of quantum programs ."""

    READOUT = "readout"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        super().__init__(protocol, options, NoiseProtocolType.POSTPROCESSING)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> PostProcessingNoise:
        return cls(d["protocol"], **d["options"])


def apply_noise(noise: Noise, samples: list[Counter]) -> list[Counter]:
    """Apply noise to samples."""
    error_fn = noise.get_noise_fn()
    # Get the number of qubits from the sample keys.
    n_qubits = len(list(samples[0].keys())[0])
    # Get the number of shots from the sample values.
    n_shots = sum(samples[0].values())
    noisy_samples: list = error_fn(
        counters=samples, n_qubits=n_qubits, options=noise.options, n_shots=n_shots
    )
    return noisy_samples
