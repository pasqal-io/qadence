from __future__ import annotations

from collections import Counter
from enum import Enum
from itertools import chain
from typing import Any

import numpy as np
import torch
from torch.distributions import normal, poisson, uniform

from qadence.logger import get_logger

logger = get_logger(__name__)


class WhiteNoise(Enum):
    """White noise distributions."""

    UNIFORM = staticmethod(uniform.Uniform(low=0.0, high=1.0))
    "Uniform white noise."

    GAUSSIAN = staticmethod(normal.Normal(loc=0.0, scale=1.0))
    "Gaussian white noise."

    POISSON = staticmethod(poisson.Poisson(rate=0.1))
    "Poisson white noise."


def bitstring_to_array(bitstring: str) -> np.array:
    return np.array([int(i) for i in bitstring])


def array_to_bitstring(bitstring: np.array) -> str:
    return "".join(([str(i) for i in bitstring]))


def bit_flip(qubit: int) -> int:
    return 1 if qubit == 0 else 0


def bs_corruption(
    bitstring: str,
    shots: int,
    fidelity: float,
    n_qubits: int,
    noise_distribution: Enum = WhiteNoise.UNIFORM,
) -> list:
    # the noise_matrix should be available to the user if they want to do error correction
    noise_matrix = noise_distribution.sample([shots, n_qubits])  # type: ignore[attr-defined]

    # simplest approach - en event occurs if its probability is higher than expected
    # by random chance
    err_idx = torch.nonzero((noise_matrix < fidelity), as_tuple=True)[1]
    all_bitstrings = [bitstring] * (shots - len(err_idx))  # add the majority correct bit strings

    def func_distort(idx: int) -> str:
        bitstring_copy = bitstring_to_array(bitstring)
        bitstring_copy[idx] = bit_flip(bitstring_copy[idx])
        return array_to_bitstring(bitstring_copy)

    all_bitstrings.extend([func_distort(idx) for idx in err_idx])
    return all_bitstrings


def readout_error(
    counters: Counter,
    n_qubits: int,
    shots: int = 1000,
    seed: int | None = None,
    fidelity: float = 0.1,
    noise_distribution: Enum = WhiteNoise.UNIFORM,
) -> list[Counter[Any]]:
    # option for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    return [
        Counter(
            chain(
                *[
                    bs_corruption(
                        bitstring=bitstring,
                        shots=shots,
                        fidelity=fidelity,
                        noise_distribution=noise_distribution,
                        n_qubits=n_qubits,
                    )
                    for bitstring, shots in counter.items()
                ]
            )
        )
        for counter in counters
    ]
