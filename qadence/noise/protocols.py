from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Counter, cast

import pyqtorch as pyq

from qadence.types import NoiseProtocolType

digital_noise_protocols = set([pyq.NoiseType(noise.value) for noise in pyq.NoiseType])
Postprocessing_PROTOCOL_TO_MODULE = {
    "readout": "qadence.noise.readout",
}


@dataclass
class NoiseProtocol:
    """Generic class for protocols."""

    def __init__(self, protocol: str, type: str = "", options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options
        self.type = type

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "type": self.type, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> NoiseProtocol:
        return cls(d["protocol"], d["type"], **d["options"])

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


@dataclass
class PulseNoise(NoiseProtocol):
    """Pulse noise is pulser-compatible noise where the right options.

    are created for a SimConfig object.
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
class PostProcessingNoise(NoiseProtocol):
    """PostProcessingNoise alters the returned output of quantum programs ."""

    READOUT = "readout"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        super().__init__(protocol, options, NoiseProtocolType.POSTPROCESSING)

    def get_noise_fn(self) -> Callable:
        try:
            module = importlib.import_module(Postprocessing_PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            raise ImportError(
                f"The module corresponding to the protocol {self.protocol} is not found."
            )
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> PostProcessingNoise:
        return cls(d["protocol"], **d["options"])


def apply_post_processing_noise(
    noise: PostProcessingNoise, samples: list[Counter]
) -> list[Counter]:
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


@dataclass
class DigitalNoise(pyq.NoiseProtocol):
    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"
    DEPHASING = "dephasing"
    DEPOLARIZING = "depolarizing"

    def __init__(self, protocol: str | list[str], options: dict = dict()) -> None:
        if len(digital_noise_protocols - set(protocol)) == 0:
            raise ValueError(
                "Protocol(s) are not available protocols. Currently supporting: {digital_noise_protocols}."
            )

        error_probability = options.get("error_probability", None)
        if error_probability is None:
            raise KeyError("A `error_probability` option should be passed in options.")
        target = options.get("target", None)
        if not (target is None or isinstance(target, int)):
            raise KeyError("A `noise_probs` option should be passed in options.")

        super().__init__(protocol, error_probability, target)
        self.type = NoiseProtocolType.DIGITAL

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> DigitalNoise:
        return cls(d["protocol"], **d["options"])

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))
