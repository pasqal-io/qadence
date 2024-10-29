from __future__ import annotations

import importlib
from typing import Any, Callable, Counter, cast

from qadence.types import NoiseEnum, NoiseProtocol

PROTOCOL_TO_MODULE = {
    "Readout": "qadence.noise.readout",
}


class NoiseSource:
    """A container for a single source of noise.

    Args:
        protocol: The name of the protocol. To be taken from `NoiseProtocol`
        options: A list of options defining the protocol.

    Examples:
    ```
        from qadence import NoiseProtocol, NoiseSource
        protocol = NoiseSource(NoiseProtocol.BITFLIP, {"error_probability": 0.5})
    ```
    """

    def __init__(self, protocol: NoiseEnum, options: dict = dict()) -> None:
        self.protocol: NoiseEnum = protocol
        self.options: dict = options

        self.verify_options()

    def verify_options(self) -> None:
        if self.protocol != NoiseProtocol.READOUT:
            name_mandatory_option = (
                "noise_probs"
                if isinstance(self.protocol, NoiseProtocol.ANALOG)
                else "error_probability"
            )
            noise_probs = self.options.get(name_mandatory_option, None)
            if noise_probs is None:
                error_txt = f"A `{name_mandatory_option}` option"
                error_txt += f"should be passed to {self.protocol} NoiseSource."
                raise KeyError(error_txt)

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
        }

    @classmethod
    def _from_dict(cls, d: dict | None) -> NoiseSource | None:
        if d is not None and d.get("protocol", None):
            return cls(d["protocol"], d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))

    def __repr__(self) -> str:
        return f"NoiseSource({self.protocol}, {str(self.options)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NoiseSource):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return self.protocol == other.protocol and self.options == other.options
        return False


class NoiseHandler:
    """A container for multiple sources of noise.

    Note `NoiseProtocol.ANALOG` and `NoiseProtocol.DIGITAL` sources cannot be both present.
    Also `NoiseProtocol.READOUT` can only be present once as the last noise sources, and only
    exclusively with `NoiseProtocol.DIGITAL` sources.

    Args:
        protocol: The protocol(s) applied. Can be
        options: A list of options defining the protocol.

    Examples:
    ```
        from qadence import NoiseProtocol, NoiseSource, NoiseHandler

        analog_options = {"noise_probs": 0.1}
        digital_options = {"error_probability": 0.1}
        readout_options = {"error_probability": 0.1, "seed": 0}

        # single noise sources
        analog_noise = NoiseHandler(NoiseProtocol.ANALOG.DEPOLARIZING, analog_options)
        digital_noise = NoiseHandler(NoiseProtocol.DIGITAL.DEPOLARIZING, digital_options)
        readout_noise = NoiseHandler(NoiseProtocol.READOUT, readout_options)

        # init from multiple sources

        digital_noise = NoiseSource(NoiseProtocol.DIGITAL.DEPOLARIZING, digital_options)
        readout_noise = NoiseSource(NoiseProtocol.READOUT, readout_options)
        noise_combination = NoiseHandler([digital_noise, readout_noise])

        # Appending noise sources
        bf_noise = NoiseSource(NoiseProtocol.DIGITAL.BITFLIP, digital_options)
        depo_noise = NoiseSource(NoiseProtocol.DIGITAL.DEPOLARIZING, digital_options)
        readout_noise = NoiseSource(NoiseProtocol.READOUT, readout_options)

        noise_combination = NoiseHandler(bf_noise)
        noise_combination.append([depo_noise, readout_noise])
    ```
    """

    def __init__(
        self,
        protocol: NoiseEnum | list[NoiseEnum] | NoiseSource | list[NoiseSource],
        options: dict | list[dict] = dict(),
    ) -> None:
        self.noise_sources: list = list()
        if isinstance(protocol, list) and isinstance(protocol[0], NoiseSource):
            self.noise_sources += protocol
        elif isinstance(protocol, NoiseSource):
            self.noise_sources += [protocol]
        else:
            protocol = protocol if isinstance(protocol, list) else [protocol]
            options = options if isinstance(options, list) else [options] * len(protocol)

            if len(options) != len(protocol):
                raise ValueError("Specify lists of same length when defining noises.")

            for proto, opt_proto in zip(protocol, options):
                self.noise_sources.append(NoiseSource(proto, opt_proto))  # type: ignore [arg-type]

        self.verify_noise_sources()

    def verify_noise_sources(self) -> None:
        """Make sure noise_sources are correct in terms of combinaison of noises."""

        if len(self.noise_sources) == 0:
            raise ValueError("NoiseHandler should be specified with one valid configuration.")

        protocols = [n.protocol for n in self.noise_sources]
        types = [type(p) for p in protocols]

        unique_types = set(types)
        if NoiseProtocol.DIGITAL in unique_types and NoiseProtocol.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both Digital and Analog noises.")

        if NoiseProtocol.ANALOG in unique_types:
            if NoiseProtocol.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and Analog noises.")
            if types.count(NoiseProtocol.ANALOG) > 1:
                raise ValueError("Multiple Analog NoiseSources are not supported yet.")

        if NoiseProtocol.READOUT in protocols:
            if protocols[-1] != NoiseProtocol.READOUT or protocols.count(NoiseProtocol.READOUT) > 1:
                raise ValueError(
                    "Only define a NoiseHandler with one READOUT as the last NoiseSource."
                )

    def __repr__(self) -> str:
        return "\n".join([str(n) for n in self.noise_sources])

    def append(
        self, other: NoiseSource | NoiseHandler | list[NoiseSource] | list[NoiseHandler]
    ) -> None:
        """Append noises to noise_sources.

        Args:
            other (NoiseSource | NoiseHandler): The noises to add.
        """
        # To avoid overwriting the noise_sources list if an error is raised, make a copy
        noises = self.noise_sources[:]
        other_list = other if isinstance(other, list) else [other]
        for noise in other_list:
            if isinstance(noise, NoiseSource):
                noises.append(noise)
            else:
                noises += noise.noise_sources
        # init may raise an error
        temp_handler = NoiseHandler(noises)
        # if verify passes, replace noise_sources
        self.noise_sources = temp_handler.noise_sources

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NoiseHandler):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            return all([n1 == n2 for n1, n2 in zip(self.noise_sources, other.noise_sources)])
        return False

    def _to_dict(self) -> dict:
        return {
            "protocol": [n.protocol for n in self.noise_sources],
            "options": [n.options for n in self.noise_sources],
        }

    @classmethod
    def _from_dict(cls, d: dict | None) -> NoiseHandler | None:
        if d is not None and d.get("protocol", None):
            return cls(d["protocol"], d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))

    def filter(self, protocol_type: NoiseEnum) -> NoiseHandler | None:
        list_noises = [n for n in self.noise_sources if isinstance(n.protocol, protocol_type)]  # type: ignore [arg-type]
        return NoiseHandler(list_noises) if len(list_noises) > 0 else None

    @classmethod
    def bitflip(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.BITFLIP, *args, **kwargs)

    @classmethod
    def phaseflip(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.PHASEFLIP, *args, **kwargs)

    @classmethod
    def digital_depolarizing(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.DEPOLARIZING, *args, **kwargs)

    @classmethod
    def pauli_channel(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.PAULI_CHANNEL, *args, **kwargs)

    @classmethod
    def amplitude_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, *args, **kwargs)

    @classmethod
    def phase_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.PHASE_DAMPING, *args, **kwargs)

    @classmethod
    def generalized_amplitude_damping(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.DIGITAL.GENERALIZED_AMPLITUDE_DAMPING, *args, **kwargs)

    @classmethod
    def analog_depolarizing(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.ANALOG.DEPOLARIZING, *args, **kwargs)

    @classmethod
    def dephasing(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.ANALOG.DEPHASING, *args, **kwargs)

    @classmethod
    def readout(cls, *args: Any, **kwargs: Any) -> NoiseHandler:
        return cls(NoiseProtocol.READOUT, *args, **kwargs)


def apply_readout_noise(noise: NoiseHandler, samples: list[Counter]) -> list[Counter]:
    """Apply readout noise to samples if provided.

    Args:
        noise (NoiseHandler): Noise to apply.
        samples (list[Counter]): Samples to alter

    Returns:
        list[Counter]: Altered samples.
    """
    readout = noise.noise_sources[-1]
    if readout.protocol == NoiseProtocol.READOUT:
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
