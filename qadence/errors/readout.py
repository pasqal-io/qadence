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
    """A helper function to convert bit strings to numpy arrays."""
    return np.array([int(i) for i in bitstring])


def array_to_bitstring(bitstring: np.array) -> str:
    """A helper function to convert numpy arrays to bit strings."""
    return "".join(([str(i) for i in bitstring]))


def bit_flip(qubit: int) -> int:
    """A helper function that reverses the states 0 and 1 in the bit string."""
    return 1 if qubit == 0 else 0


def bs_corruption(
    bitstring: str,
    n_shots: int,
    error_probability: float,
    n_qubits: int,
    noise_distribution: Enum = WhiteNoise.UNIFORM,
) -> list:
    # the noise_matrix should be available to the user if they want to do error correction
    noise_matrix = noise_distribution.sample([n_shots, n_qubits])  # type: ignore[attr-defined]

    # simplest approach - en event occurs if its probability is higher than expected
    # by random chance
    err_idx = torch.nonzero((noise_matrix < error_probability), as_tuple=True)[1]
    all_bitstrings = [bitstring] * (n_shots - len(err_idx))  # add the majority correct bit strings

    def func_distort(idx: int) -> str:
        bitstring_copy = bitstring_to_array(bitstring)
        bitstring_copy[idx] = bit_flip(bitstring_copy[idx])
        return array_to_bitstring(bitstring_copy)

    all_bitstrings.extend([func_distort(idx) for idx in err_idx])
    return all_bitstrings


def error(
    counters: Counter,
    n_qubits: int,
    options: dict = dict(),
) -> list[Counter[Any]]:
    """
    Implements a simple uniform readout error model for position-independent bit string
    corruption.

    Args:
        counters: Samples of bit string as Counters.
        n_qubits: Number of shots to sample.
        seed: Random seed value if any.
        error_probability: Uniform error probability of wrong readout at any position
        in the bit strings.
        noise_distribution: Noise distribution.

    Returns:
        Samples of corrupted bit strings as list[Counter].
    """

    seed = options.get("seed")
    error_probability = options.get("error_probability", 0.1)
    noise_distribution = options.get("noise_distribution", WhiteNoise.UNIFORM)

    # option for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    corrupted_bitstrings = []
    for counter in counters:
        corrupted_bitstrings.append(
            Counter(
                chain(
                    *[
                        bs_corruption(
                            bitstring=bitstring,
                            n_shots=n_shots,
                            error_probability=error_probability,
                            noise_distribution=noise_distribution,
                            n_qubits=n_qubits,
                        )
                        for bitstring, n_shots in counter.items()
                    ]
                )
            )
        )
    return corrupted_bitstrings
