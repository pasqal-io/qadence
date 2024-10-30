from __future__ import annotations

import importlib
from itertools import compress
from typing import Any, Callable, Counter, cast

from qadence.types import NoiseEnum, NoiseProtocol

PROTOCOL_TO_MODULE = {
    "Readout": "qadence.noise.readout",
}


class NoiseHandler:
    """A container for multiple sources of noise.

    Note `NoiseProtocol.ANALOG` and `NoiseProtocol.DIGITAL` sources cannot be both present.
    Also `NoiseProtocol.READOUT` can only be present once as the last noise sources, and only
    exclusively with `NoiseProtocol.DIGITAL` sources.

    Args:
        protocol: The protocol(s) applied. To be defined from `NoiseProtocol`.
        options: A list of options defining the protocol.
            For `NoiseProtocol.ANALOG`, options should contain a field `noise_probs`.
            For `NoiseProtocol.DIGITAL`, options should contain a field `error_probability`.

    Examples:
    ```
        from qadence import NoiseProtocol, NoiseHandler

        analog_options = {"noise_probs": 0.1}
        digital_options = {"error_probability": 0.1}
        readout_options = {"error_probability": 0.1, "seed": 0}

        # single noise sources
        analog_noise = NoiseHandler(NoiseProtocol.ANALOG.DEPOLARIZING, analog_options)
        digital_depo_noise = NoiseHandler(NoiseProtocol.DIGITAL.DEPOLARIZING, digital_options)
        readout_noise = NoiseHandler(NoiseProtocol.READOUT, readout_options)

        # init from multiple sources
        protocols: list = [NoiseProtocol.DIGITAL.DEPOLARIZING, NoiseProtocol.READOUT]
        options: list = [digital_options, readout_noise]
        noise_combination = NoiseHandler(protocols, options)

        # Appending noise sources
        noise_combination = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, digital_options)
        noise_combination.append([digital_depo_noise, readout_noise])
    ```
    """

    def __init__(
        self,
        protocol: NoiseEnum | list[NoiseEnum],
        options: dict | list[dict] = dict(),
    ) -> None:
        self.protocol = protocol if isinstance(protocol, list) else [protocol]
        self.options = options if isinstance(options, list) else [options] * len(self.protocol)
        self.verify_all_protocols()

    def _verify_single_protocol(self, protocol: NoiseEnum, option: dict) -> None:
        if protocol != NoiseProtocol.READOUT:
            name_mandatory_option = (
                "noise_probs" if isinstance(protocol, NoiseProtocol.ANALOG) else "error_probability"
            )
            noise_probs = option.get(name_mandatory_option, None)
            if noise_probs is None:
                error_txt = f"A `{name_mandatory_option}` option"
                error_txt += f"should be passed for protocol {protocol}."
                raise KeyError(error_txt)

    def verify_all_protocols(self) -> None:
        """Make sure all protocols are correct in terms and their combination too."""

        if len(self.protocol) == 0:
            raise ValueError("NoiseHandler should be specified with one valid configuration.")

        if len(self.protocol) != len(self.options):
            raise ValueError("Specify lists of same length when defining noises.")

        for protocol, option in zip(self.protocol, self.options):
            self._verify_single_protocol(protocol, option)

        types = [type(p) for p in self.protocol]
        unique_types = set(types)
        if NoiseProtocol.DIGITAL in unique_types and NoiseProtocol.ANALOG in unique_types:
            raise ValueError("Cannot define a config with both Digital and Analog noises.")

        if NoiseProtocol.ANALOG in unique_types:
            if NoiseProtocol.READOUT in unique_types:
                raise ValueError("Cannot define a config with both READOUT and Analog noises.")
            if types.count(NoiseProtocol.ANALOG) > 1:
                raise ValueError("Multiple Analog Noises are not supported yet.")

        if NoiseProtocol.READOUT in self.protocol:
            if (
                self.protocol[-1] != NoiseProtocol.READOUT
                or self.protocol.count(NoiseProtocol.READOUT) > 1
            ):
                raise ValueError("Only define a NoiseHandler with one READOUT as the last Noise.")

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"Noise({protocol}, {str(option)})"
                for protocol, option in zip(self.protocol, self.options)
            ]
        )

    def get_noise_fn(self, index_protocol: int) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol[index_protocol]])
        except KeyError:
            ImportError(
                f"The module for the protocol {self.protocol[index_protocol]} is not found."
            )
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def append(self, other: NoiseHandler | list[NoiseHandler]) -> None:
        """Append noises.

        Args:
            other (NoiseHandler | list[NoiseHandler]): The noises to add.
        """
        # To avoid overwriting the noise_sources list if an error is raised, make a copy
        other_list = other if isinstance(other, list) else [other]
        protocols = self.protocol[:]
        options = self.options[:]

        for noise in other_list:
            protocols += noise.protocol
            options += noise.options

        # init may raise an error
        temp_handler = NoiseHandler(protocols, options)
        # if verify passes, replace protocols and options
        self.protocol = temp_handler.protocol
        self.options = temp_handler.options

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NoiseHandler):
            raise TypeError(f"Cant compare {type(self)} to {type(other)}")
        if isinstance(other, type(self)):
            protocols_equal = all([p1 == p2 for p1, p2 in zip(self.protocol, other.protocol)])
            options_equal = all([o1 == o2 for o1, o2 in zip(self.options, other.options)])
            return protocols_equal and options_equal

        return False

    def _to_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "options": self.options,
        }

    @classmethod
    def _from_dict(cls, d: dict | None) -> NoiseHandler | None:
        if d is not None and d.get("protocol", None):
            return cls(d["protocol"], d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))

    def filter(self, protocol: NoiseEnum) -> NoiseHandler | None:
        is_protocol: list = [isinstance(p, protocol) for p in self.protocol]  # type: ignore[arg-type]
        return (
            NoiseHandler(
                list(compress(self.protocol, is_protocol)),
                list(compress(self.options, is_protocol)),
            )
            if len(is_protocol) > 0
            else None
        )

    def bitflip(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, *args, **kwargs))
        return self

    def phaseflip(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.PHASEFLIP, *args, **kwargs))
        return self

    def digital_depolarizing(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.DEPOLARIZING, *args, **kwargs))
        return self

    def pauli_channel(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.PAULI_CHANNEL, *args, **kwargs))
        return self

    def amplitude_damping(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, *args, **kwargs))
        return self

    def phase_damping(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.DIGITAL.PHASE_DAMPING, *args, **kwargs))
        return self

    def generalized_amplitude_damping(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(
            NoiseHandler(NoiseProtocol.DIGITAL.GENERALIZED_AMPLITUDE_DAMPING, *args, **kwargs)
        )
        return self

    def analog_depolarizing(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.ANALOG.DEPOLARIZING, *args, **kwargs))
        return self

    def dephasing(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.ANALOG.DEPHASING, *args, **kwargs))
        return self

    def readout(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.READOUT, *args, **kwargs))
        return self


def apply_readout_noise(noise: NoiseHandler, samples: list[Counter]) -> list[Counter]:
    """Apply readout noise to samples if provided.

    Args:
        noise (NoiseHandler): Noise to apply.
        samples (list[Counter]): Samples to alter

    Returns:
        list[Counter]: Altered samples.
    """
    if noise.protocol[-1] == NoiseProtocol.READOUT:
        error_fn = noise.get_noise_fn(-1)
        # Get the number of qubits from the sample keys.
        n_qubits = len(list(samples[0].keys())[0])
        # Get the number of shots from the sample values.
        n_shots = sum(samples[0].values())
        noisy_samples: list = error_fn(
            counters=samples, n_qubits=n_qubits, options=noise.options[-1], n_shots=n_shots
        )
        return noisy_samples
    else:
        return samples
