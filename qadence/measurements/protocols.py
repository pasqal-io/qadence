from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, cast

PROTOCOL_TO_MODULE = {
    "shadow": "qadence.measurements.shadow",
    "tomography": "qadence.measurements.tomography",
}


# TODO: make this a StrEnum to keep consistency with the rest of the interface
@dataclass
class Measurements:
    SHADOW = "shadow"
    TOMOGRAPHY = "tomography"

    def __init__(self, protocol: str, options: dict) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def get_measurement_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "compute_expectation")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> Measurements | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None
