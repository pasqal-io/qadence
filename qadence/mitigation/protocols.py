from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, cast

PROTOCOL_TO_MODULE = {
    "zne": "qadence.mitigation.zne",
}


@dataclass
class ErrorMitigation:
    ZNE = "zne"

    def __init__(self, protocol: str, options: dict) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def get_errormitigation_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "mitigate")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> ErrorMitigation | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None
