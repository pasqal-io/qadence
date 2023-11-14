from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable, cast

from qadence.noise.protocols import Noise

PROTOCOL_TO_MODULE = {
    "readout": "qadence.mitigations.readout",
    "zne": "qadence.mitigations.zne",
}


@dataclass
class Mitigations:
    READOUT = "readout"
    ZNE = "zne"

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


def apply_mitigation(
    noise: Noise, mitigation: Mitigations, samples: list[Counter]
) -> list[Counter]:
    """Apply mitigation to samples."""
    mitigation_fn = mitigation.get_mitigation_fn()
    # breakpoint()
    mitigated_samples: list[Counter] = mitigation_fn(
        noise=noise, mitigation=mitigation, samples=samples
    )
    return mitigated_samples
