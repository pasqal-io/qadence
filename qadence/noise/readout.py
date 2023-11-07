from __future__ import annotations

from collections import Counter
from enum import Enum
from itertools import chain

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
    """
    A helper function to convert bit strings to numpy arrays.

    Args:
        bitstring:  A str format of a bit string.

    Returns:
        A numpy array out of the input bit string.
    """
    return np.array(list(bitstring)).astype(int)


def array_to_bitstring(bitstring: np.array) -> str:
    """
    A helper function to convert numpy arrays to bit strings.

    Args:
        bitstring: A numpy array format of a bit string.

    Returns:
        A str out of the input bit string.
    """
    return "".join(bitstring.astype("str"))


def bit_flip(bit: int) -> int:
    """
    A helper function that reverses the states 0 and 1 in the bit string.

    Args:
        bit: A integer-value bit in a bitstring to be inverted.

    Returns:
        The inverse value of the input bit
    """
    return 1 if bit == 0 else 0


def sample_to_matrix(sample: dict) -> np.array:
    """
    A helper function that maps a sample dict to a bit string array.

    Args:
        sample: A dictionary with bit stings as keys and values
        as their counts.

    Returns: A numpy array (matrix) of bit strings n_shots x n_qubits.
    """
    return np.array(
        [
            i
            for i in chain.from_iterable(
                [sample[bitstring] * [bitstring_to_array(bitstring)] for bitstring in sample.keys()]
            )
        ]
    )


def create_noise_matrix(
    noise_distribution: torch.distributions, n_shots: int, n_qubits: int
) -> np.array:
    """
    A helper function that creates a noise matrix for bit string corruption.
    NB: The noise matrix is not square, as all bits are considered independent.

    Args:
        noise_distribution: Torch statistical distribution one of Gaussian,
        Uniform, of Poisson.
        n_shots: Number of shots/samples.
        n_qubits: Number of qubits

    Returns:
        A sample out of the requested distribution given the number of shots/samples.
    """
    # the noise_matrix should be available to the user if they want to do error correction
    return noise_distribution.sample([n_shots, n_qubits])


def bs_corruption(
    n_shots: int,
    n_qubits: int,
    err_idx: list,
    sample: np.array,
) -> Counter:
    """
    A function that incorporates the expected readout error in a sample of bit strings
    given a noise matrix.

    Args:
        n_qubits: Number of qubits in the bit string.
        n_shots: Number of shots to sample.
        err_idx: A Boolean array of bit string indices that need to be corrupted.
        sample: A matrix of bit strings n_shots x n_qubits.

    Returns:
        A counter of bit strings after readout corruption.
    """

    def vflip(sample: int, err_idx: bool, idx: int) -> np.array:
        if err_idx:
            return bit_flip(sample)
        else:
            return sample

    vbit_ind = np.vectorize(vflip)

    return Counter(
        [
            "".join(k)
            for k in map(
                lambda i: vbit_ind(sample[i], err_idx[i], range(n_qubits)).astype(str).tolist(),
                range(n_shots),
            )
        ]
    )


def error(
    counters: list[Counter],
    n_qubits: int,
    n_shots: int = 1000,
    options: dict = dict(),
) -> list[Counter]:
    """
    Implements a simple uniform readout error model for position-independent bit string
    corruption.

    Args:
        counters (list): Samples of bit string as Counters.
        n_qubits: Number of qubits in the bit string.
        n_shots: Number of shots to sample.
        options: A dict of options:
          seed: Random seed value if any.
          error_probability: Uniform error probability of wrong readout at any position
            in the bit strings.
          noise_distribution: Noise distribution.
          noise_matrix: An input noise matrix if known.

    Returns:
        Samples of corrupted bit strings as list[Counter].
    """

    seed = options.get("seed", None)
    error_probability = options.get("error_probability", 0.1)
    noise_distribution = options.get("noise_distribution", WhiteNoise.UNIFORM)
    noise_matrix = options.get("noise_matrix")

    # option for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    if noise_matrix is None:
        # assumes that all bits can be flipped independently of each other
        noise_matrix = create_noise_matrix(noise_distribution, n_shots, n_qubits)
    else:
        # check noise_matrix shape and values
        assert (
            noise_matrix.shape[0] == noise_matrix.shape[1]
        ), "The error probabilities matrix needs to be square."
        assert noise_matrix.shape == (
            n_qubits,
            n_qubits,
        ), "The error probabilities matrix needs to be n_qubits x n_qubits."

    # the simplest approach - an event occurs if its probability is higher than expected
    # by random chance
    err_idx = np.array([(item).numpy() for i, item in enumerate(noise_matrix < error_probability)])

    corrupted_bitstrings = []
    for counter in counters:
        sample = sample_to_matrix(counter)
        corrupted_bitstrings.append(
            bs_corruption(n_shots=n_shots, err_idx=err_idx, sample=sample, n_qubits=n_qubits)
        )
    return corrupted_bitstrings
