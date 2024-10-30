from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable, cast

from qadence.noise.protocols import NoiseHandler

PROTOCOL_TO_MODULE = {
    "readout": "qadence.mitigations.readout",
    "zne": "qadence.mitigations.analog_zne",
}


@dataclass
class Mitigations:
    READOUT = "readout"
    ANALOG_ZNE = "zne"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def get_mitigation_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "mitigate")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> Mitigations | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


def apply_mitigation(
    noise: NoiseHandler, mitigation: Mitigations, samples: list[Counter]
) -> list[Counter]:
    """Apply mitigation to samples."""
    mitigation_fn = mitigation.get_mitigation_fn()
    mitigated_samples: list[Counter] = mitigation_fn(
        noise=noise, mitigation=mitigation, samples=samples
    )
    return mitigated_samples
