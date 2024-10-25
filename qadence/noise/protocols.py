from __future__ import annotations

import importlib
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

    def __init__(self, protocol: str, options: dict = dict(), noise_type: str = "") -> None:
        if protocol not in supported_noise_protocols:
            raise ValueError(
                "Protocol {protocol} is not supported. Choose from {supported_noise_protocols}."
            )

        self.protocol: str = protocol

        self.options: dict = options

        self.noise_type: str = noise_type

        # forcing in certain cases the type of predefined protocols
        if self.noise_type == "":
            if protocol == "Readout":
                self.noise_type = NoiseProtocolType.READOUT
            if self.protocol == "Dephasing":
                self.noise_type = NoiseProtocolType.ANALOG
            if self.protocol in digital_noise_protocols:
                self.noise_type = NoiseProtocolType.DIGITAL
        else:
            if self.noise_type not in [NoiseProtocolType(t.value) for t in NoiseProtocolType]:
                raise ValueError("Noise type {self.noise_type} is not supported.")

        self.verify_options()

    def verify_options(self) -> None:
        if self.noise_type != NoiseProtocolType.READOUT:
            name_mandatory_option = (
                "noise_probs"
                if self.noise_type == NoiseProtocolType.ANALOG
                else "error_probability"
            )
            noise_probs = self.options.get(name_mandatory_option, None)
            if noise_probs is None:
                KeyError(
                    "A `{name_mandatory_option}` option should be passed \
                      to the NoiseSource of type {self.noise_type}."
                )

    def get_noise_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options, "noise_type": self.noise_type}

    @classmethod
    def _from_dict(cls, d: dict) -> NoiseSource | None:
        if d:
            noise_type = d.get("noise_type", "")
            return cls(d["protocol"], **d["options"], noise_type=noise_type)
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


class NoiseHandler:
    """A container for multiple sources of noise."""

    def __init__(
        self,
        protocol: str | NoiseSource | list[str] | list[NoiseSource],
        options: dict | list[dict] = dict(),
        noise_type: str | list[str] = "",
    ) -> None:
        self.noise_sources: list = list()
        if isinstance(protocol, list) and isinstance(protocol[0], NoiseSource):
            self.noise_sources += protocol
        elif isinstance(protocol, NoiseSource):
            self.noise_sources += [protocol]
        else:
            protocol = [protocol] if isinstance(protocol, str) else protocol
            options = [options] * len(protocol) if isinstance(options, dict) else options
            types = [noise_type] * len(protocol) if isinstance(noise_type, str) else noise_type

            if len(options) != len(protocol) or len(noise_type) != len(protocol):
                raise ValueError("Specify lists of same length when defining noises.")

            for proto, opt_proto, type_proto in zip(protocol, options, types):
                self.noise_sources.append(NoiseSource(proto, opt_proto, type_proto))  # type: ignore [arg-type]

        types = [n.noise_type for n in self.noise_sources]
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
                    "Only define a NoiseHandler with one READOUT as the last NoiseSource."
                )

    def _to_dict(self) -> dict:
        return {
            "protocol": [n.protocol for n in self.noise_sources],
            "options": [n.options for n in self.noise_sources],
            "noise_type": [n.noise_type for n in self.noise_sources],
        }

    @classmethod
    def _from_dict(cls, d: dict) -> NoiseHandler | None:
        if d:
            noise_type = d.get("noise_type", "")
            return cls(d["protocol"], **d["options"], noise_type=noise_type)
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


class DigitalNoiseConfig(NoiseHandler):
    def __init__(
        self,
        protocol: str | NoiseSource | list[str] | list[NoiseSource],
        options: dict | list[dict] = dict(),
        type: str | list[str] = "",
    ) -> None:
        super().__init__(protocol, options, type)
        types = [n.noise_type for n in self.noise_sources]


def apply_readout_noise(noise: NoiseHandler, samples: list[Counter]) -> list[Counter]:
    """Apply readout noise to samples if provided.

    Args:
        noise (NoiseHandler): Noise to apply.
        samples (list[Counter]): Samples to alter

    Returns:
        list[Counter]: Altered samples.
    """
    readout = noise.noise_sources[-1]
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
