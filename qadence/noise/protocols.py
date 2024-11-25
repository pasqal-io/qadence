from __future__ import annotations

from itertools import compress
from typing import Any

from qadence.types import NoiseEnum, NoiseProtocol


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
        if not isinstance(protocol, NoiseProtocol.READOUT):  # type: ignore[arg-type]
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

        if NoiseProtocol.READOUT in unique_types:
            if (
                not isinstance(self.protocol[-1], NoiseProtocol.READOUT)
                or types.count(NoiseProtocol.READOUT) > 1
            ):
                raise ValueError("Only define a NoiseHandler with one READOUT as the last Noise.")

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"Noise({protocol}, {str(option)})"
                for protocol, option in zip(self.protocol, self.options)
            ]
        )

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
        protocol_matches: list = [isinstance(p, protocol) for p in self.protocol]  # type: ignore[arg-type]

        # if we have at least a match
        if True in protocol_matches:
            return NoiseHandler(
                list(compress(self.protocol, protocol_matches)),
                list(compress(self.options, protocol_matches)),
            )
        return None

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

    def readout_independent(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.READOUT.INDEPENDENT, *args, **kwargs))
        return self

    def readout_correlated(self, *args: Any, **kwargs: Any) -> NoiseHandler:
        self.append(NoiseHandler(NoiseProtocol.READOUT.CORRELATED, *args, **kwargs))
        return self
