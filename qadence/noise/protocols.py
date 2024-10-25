from __future__ import annotations

import importlib
from typing import Any, Callable, Counter, cast

from qadence.types import DigitalNoiseType, NoiseProtocolType, NoiseType

PROTOCOL_TO_MODULE = {
    "Readout": "qadence.noise.readout",
}

# Temporary solution
digital_noise_protocols = set(DigitalNoiseType.list())
supported_noise_protocols = NoiseType.list()


class NoiseSource:
    """A container for a single source of noise."""

    def __init__(self, protocol: str, options: dict = dict(), protocol_type: str = "") -> None:
        if protocol not in supported_noise_protocols:
            raise ValueError(
                "Protocol {protocol} is not supported. Choose from {supported_noise_protocols}."
            )

        self.protocol: str = protocol

        self.options: dict = options

        self.protocol_type: str = protocol_type

        # forcing in certain cases the type of predefined protocols
        if self.protocol_type == "":
            if protocol == "Readout":
                self.protocol_type = NoiseProtocolType.READOUT
            if self.protocol == "Dephasing":
                self.protocol_type = NoiseProtocolType.ANALOG
            if self.protocol in digital_noise_protocols:
                self.protocol_type = NoiseProtocolType.DIGITAL
        else:
            if self.protocol_type not in [NoiseProtocolType(t.value) for t in NoiseProtocolType]:
                raise ValueError("Noise type {self.protocol_type} is not supported.")

        self.verify_options()

    def verify_options(self) -> None:
        if self.protocol_type != NoiseProtocolType.READOUT:
            name_mandatory_option = (
                "noise_probs"
                if self.protocol_type == NoiseProtocolType.ANALOG
                else "error_probability"
            )
            noise_probs = self.options.get(name_mandatory_option, None)
            if noise_probs is None:
                KeyError(
                    "A `{name_mandatory_option}` option should be passed \
                      to the NoiseSource of type {self.protocol_type}."
                )

    def get_noise_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "options": self.options,
            "protocol_type": self.protocol_type,
        }

    @classmethod
    def _from_dict(cls, d: dict | None) -> NoiseSource | None:
        if d:
            protocol_type = d.get("protocol_type", "")
            return cls(d["protocol"], **d["options"], protocol_type=protocol_type)
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
        protocol_type: str | list[str] = "",
    ) -> None:
        self.noise_sources: list = list()
        if isinstance(protocol, list) and isinstance(protocol[0], NoiseSource):
            self.noise_sources += protocol
        elif isinstance(protocol, NoiseSource):
            self.noise_sources += [protocol]
        else:
            protocol = protocol if isinstance(protocol, list) else [protocol]
            options = options if isinstance(options, list) else [options] * len(protocol)
            types = (
                protocol_type
                if isinstance(protocol_type, list)
                else [protocol_type] * len(protocol)
            )

            if len(options) != len(protocol) or len(types) != len(protocol):
                raise ValueError("Specify lists of same length when defining noises.")

            for proto, opt_proto, type_proto in zip(protocol, options, types):
                self.noise_sources.append(NoiseSource(proto, opt_proto, type_proto))  # type: ignore [arg-type]

        if len(self.noise_sources) == 0:
            raise ValueError("NoiseHandler should be specified with one valid configuration.")

        types = [n.protocol_type for n in self.noise_sources]

        unique_types = set(types)
        if NoiseProtocolType.DIGITAL in unique_types and NoiseProtocolType.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both Digital and Analog noises.")

        if NoiseProtocolType.ANALOG in unique_types:
            if NoiseProtocolType.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and Analog noises.")
            if types.count(NoiseProtocolType.ANALOG) > 1:
                raise ValueError("Multiple Analog NoiseSources are not supported yet.")

        if NoiseProtocolType.READOUT in unique_types:
            if types[-1] != NoiseProtocolType.READOUT or types.count(NoiseProtocolType.READOUT) > 1:
                raise ValueError(
                    "Only define a NoiseHandler with one READOUT as the last NoiseSource."
                )

    def _to_dict(self) -> dict:
        return {
            "protocol": [n.protocol for n in self.noise_sources],
            "options": [n.options for n in self.noise_sources],
            "protocol_type": [n.protocol_type for n in self.noise_sources],
        }

    @classmethod
    def _from_dict(cls, d: dict | None) -> NoiseHandler | None:
        if d:
            protocol_type = d.get("protocol_type", "")
            return cls(d["protocol"], **d["options"], protocol_type=protocol_type)
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))

    def filter(self, protocol_type: str) -> NoiseHandler | None:
        list_noises = [
            n for n in self.noise_sources if n.protocol_type == NoiseProtocolType(protocol_type)
        ]
        return NoiseHandler(list_noises) if len(list_noises) > 0 else None

    @classmethod
    def bitflip(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.BITFLIP, *args, **kwargs)

    @classmethod
    def phaseflip(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.PHASEFLIP, *args, **kwargs)

    @classmethod
    def depolarizing(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.DEPOLARIZING, *args, **kwargs)

    @classmethod
    def pauli_channel(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.PAULI_CHANNEL, *args, **kwargs)

    @classmethod
    def amplitude_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.AMPLITUDE_DAMPING, *args, **kwargs)

    @classmethod
    def phase_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.PHASE_DAMPING, *args, **kwargs)

    @classmethod
    def generalized_amplitude_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.GENERALIZED_AMPLITUDE_DAMPING, *args, **kwargs)

    @classmethod
    def dephasing(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.DEPHASING, *args, **kwargs)

    @classmethod
    def readout(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseType.READOUT, *args, **kwargs)


def apply_readout_noise(noise: NoiseHandler, samples: list[Counter]) -> list[Counter]:
    """Apply readout noise to samples if provided.

    Args:
        noise (NoiseHandler): Noise to apply.
        samples (list[Counter]): Samples to alter

    Returns:
        list[Counter]: Altered samples.
    """
    readout = noise.noise_sources[-1]
    if readout.protocol_type == NoiseProtocolType.READOUT:
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
