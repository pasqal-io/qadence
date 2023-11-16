from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Counter, cast

PROTOCOL_TO_MODULE = {
    "readout": "qadence.noise.readout",
}


@dataclass
class Noise:
    DEPHASING = "dephasing"
    DEPOLARIZING = "depolarizing"
    READOUT = "readout"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def get_noise_fn(self) -> Callable:
        try:
            module = importlib.import_module(PROTOCOL_TO_MODULE[self.protocol])
        except KeyError:
            ImportError(f"The module corresponding to the protocol {self.protocol} is not found.")
        fn = getattr(module, "add_noise")
        return cast(Callable, fn)

    def _to_dict(self) -> dict:
        return {"protocol": self.protocol, "options": self.options}

    @classmethod
    def _from_dict(cls, d: dict) -> Noise | None:
        if d:
            return cls(d["protocol"], **d["options"])
        return None

    @classmethod
    def list(cls) -> list:
        return list(filter(lambda el: not el.startswith("__"), dir(cls)))


def apply_noise(noise: Noise, samples: list[Counter]) -> list[Counter]:
    """Apply noise to samples."""
    error_fn = noise.get_noise_fn()
    # Get the number of qubits from the sample keys.
    n_qubits = len(list(samples[0].keys())[0])
    # Get the number of shots from the sample values.
    n_shots = sum(samples[0].values())
    noisy_samples: list = error_fn(
        counters=samples, n_qubits=n_qubits, options=noise.options, n_shots=n_shots
    )
    return noisy_samples
