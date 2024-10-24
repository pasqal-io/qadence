from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Callable, Counter, cast

from pyqtorch.noise import NoiseProtocol

from qadence.types import DigitalNoiseType, NoiseProtocolType, NoiseType

PROTOCOL_TO_MODULE = {
    "Readout": "qadence.noise.readout",
}

# Temporary solution
DigitalNoise = NoiseProtocol
digital_noise_protocols = set(DigitalNoiseType.list())
supported_noise_protocols = NoiseType.list()


class NoiseSource:
    """A container for a single source of noise."""

    def __init__(self, protocol: str, options: dict = dict(), type: str = "") -> None:
        if protocol not in supported_noise_protocols:
            raise ValueError(
                "Protocol {protocol} is not supported. Choose from {supported_noise_protocols}."
            )

        self.protocol: str = protocol

        self.options: dict = options

        self.type: str = type

        # forcing in certain cases the type of predefined protocols
        if self.type == "":
            if protocol == "Readout":
                self.type = NoiseProtocolType.READOUT
            if self.protocol == "Dephasing":
                self.type = NoiseProtocolType.ANALOG
            if self.protocol in digital_noise_protocols:
                self.type = NoiseProtocolType.DIGITAL
        else:
            if self.type not in [NoiseProtocolType(t.value) for t in NoiseProtocolType]:
                raise ValueError("Noise type {self.type} is not supported.")

        self.verify_options()

    def verify_options(self) -> None:
        if self.type == NoiseProtocolType.ANALOG:
            noise_probs = self.options.get("noise_probs", None)
            if noise_probs is None:
                KeyError("A `noise probs` option should be passed to the NoiseSource.")
            if not (isinstance(noise_probs, float) or isinstance(noise_probs, Iterable)):
                KeyError(
                    "A single or a range of noise probabilities"
                    " should be passed. Got {type(noise_probs)}."
                )
        elif self.type == NoiseProtocolType.DIGITAL:
            error_prob = self.options.get("error_probability", None)
            if not (error_prob and isinstance(error_prob, float)):
                KeyError("A `error_probability` option should be passed to the NoiseSource.")
        else:
            pass

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
    def _from_dict(cls, d: dict) -> NoiseSource | None:
        if d:
            type = d.get("type", "")
            return cls(d["protocol"], **d["options"], type=type)
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


class NoiseConfig:
    """A container for multiple sources of noise."""

    def __init__(
        self,
        protocol: str | NoiseSource | list[str] | list[NoiseSource],
        options: dict | list[dict] = dict(),
        type: str | list[str] = "",
    ) -> None:
        self.noise_sources: list = list()
        if isinstance(protocol, list) and isinstance(protocol[0], NoiseSource):
            self.noise_sources += protocol
        elif isinstance(protocol, NoiseSource):
            self.noise_sources += [protocol]
        else:
            protocol = [protocol] if isinstance(protocol, str) else protocol
            options = [options] * len(protocol) if isinstance(options, dict) else options
            types = [type] * len(protocol) if isinstance(type, str) else type

            if len(options) != len(protocol) or len(types) != len(protocol):
                raise ValueError("Specify lists of same length when defining noises.")

            for proto, opt_proto, type_proto in zip(protocol, options, types):
                self.noise_sources.append(NoiseSource(proto, opt_proto, type_proto))  # type: ignore [arg-type]

        types = [n.type for n in self.noise_sources]
        unique_types = set(types)
        if NoiseProtocolType.DIGITAL in unique_types and NoiseProtocolType.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both Digital and Analog noises.")

        if NoiseProtocolType.ANALOG in unique_types:
            if NoiseProtocolType.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and Analog noises.")
            if types.count(NoiseProtocolType.ANALOG) > 1:
                raise ValueError("Multiple Analog NoiseSources are not supported.")

        if NoiseProtocolType.READOUT in unique_types:
            if types[-1] != NoiseProtocolType.READOUT or types.count(NoiseProtocolType.READOUT) > 1:
                raise ValueError(
                    "Only define a NoiseConfig with one READOUT as the last NoiseSource."
                )

    def _to_dict(self) -> dict:
        return {
            "protocol": [n.protocol for n in self.noise_sources],
            "options": [n.options for n in self.noise_sources],
            "type": [n.type for n in self.noise_sources],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> NoiseConfig | None:
        if d:
            type = d.get("type", "")
            return cls(d["protocol"], **d["options"], type=type)
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


def apply_noise(noise: NoiseSource | NoiseConfig, samples: list[Counter]) -> list[Counter]:
    """Apply readout noise to samples if provided.

    Args:
        noise (NoiseSource | NoiseConfig): Noise to apply.
        samples (list[Counter]): Samples to alter

    Returns:
        list[Counter]: Altered samples.
    """
    readout = noise if isinstance(noise, NoiseSource) else noise.noise_sources[-1]
    if readout.type == NoiseProtocolType.READOUT:
        error_fn = readout.get_noise_fn()
        # Get the number of qubits from the sample keys.
        n_qubits = len(list(samples[0].keys())[0])
        # Get the number of shots from the sample values.
        n_shots = sum(samples[0].values())
        noisy_samples: list = error_fn(
            counters=samples, n_qubits=n_qubits, options=readout.options, n_shots=n_shots
        )
        return noisy_samples
    else:
        return samples
