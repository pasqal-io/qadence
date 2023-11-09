from __future__ import annotations

from collections import Counter
from enum import Enum

import numpy as np
import torch
from torch import Tensor
from torch.distributions import normal, poisson, uniform

from qadence.logger import get_logger

logger = get_logger(__name__)


class WhiteNoise(Enum):
    """White noise distributions."""

    UNIFORM = staticmethod(uniform.Uniform(low=0.0, high=1.0))
    """Uniform white noise."""

    GAUSSIAN = staticmethod(normal.Normal(loc=0.0, scale=1.0))
    """Gaussian white noise."""

    POISSON = staticmethod(poisson.Poisson(rate=0.1))
    """Poisson white noise."""


def bitstring_to_tensor(bitstring: str, output_type: str = "torch") -> np.array | Tensor:
    """
    A helper function to convert bit strings to torch.Tensor or numpy.array.

    Args:
        bitstring:  A str format of a bit string.
        output_type: A str torch | numpy for the type of the output.
        Default torch.

    Returns:
        A torch.Tensor or np.array out of the input bit string.
    """
    return (
        torch.as_tensor(list(map(int, bitstring)))
        if output_type == "torch"
        else np.array(list(bitstring)).astype(int)
    )


def tensor_to_bitstring(bitstring: Tensor | np.array, output_type: str = "torch") -> str:
    """
    A helper function to convert torch.Tensor or numpy.array to bit strings.

    Args:
        bitstring: A torch.Tensor or numpy.array format of a bit string.

    Returns:
        A str out of the input bit string.
    """
    return (
        "".join(list(map(str, bitstring.detach().tolist())))
        if output_type == "torch"
        else "".join(bitstring.astype("str"))
    )


def bit_flip(bit: Tensor, cond: Tensor) -> Tensor:
    """
    A helper function that reverses the states 0 and 1 in the bit string.

    Args:
        bit: A integer-value bit in a bitstring to be inverted.
        cond: A Bool value of whether or not a bit should be flipped.

    Returns:
        The inverse value of the input bit
    """
    return torch.where(cond, torch.where(bit == 0, 1, 0), bit)


def sample_to_matrix(sample: dict) -> Tensor:
    """
    A helper function that maps a sample dict to a bit string array.

    Args:
        sample: A dictionary with bit stings as keys and values
        as their counts.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

    return torch.concatenate(
        list(
            map(
                lambda bitstring: torch.broadcast_to(
                    bitstring_to_tensor(bitstring), [sample[bitstring], len(bitstring)]
                ),
                sample.keys(),
            )
        )
    )


def create_noise_matrix(
    noise_distribution: torch.distributions, n_shots: int, n_qubits: int
) -> Tensor:
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
    err_idx: Tensor,
    sample: Tensor,
) -> Counter:
    """
    A function that incorporates the expected readout error in a sample of bit strings.

    given a noise matrix.

    Args:
        err_idx: A Boolean array of bit string indices that need to be corrupted.
        sample: A torch.Tensor of bit strings n_shots x n_qubits.

    Returns:
        A counter of bit strings after readout corruption.
    """

    func = torch.func.vmap(bit_flip)

    return Counter([tensor_to_bitstring(k) for k in func(sample, err_idx)])


def add_noise(
    counters: list[Counter],
    n_qubits: int,
    n_shots: int = 1000,
    options: dict = dict(),
) -> list[Counter]:
    """
    Implements a simple uniform readout error model for position-independent bit string.

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
    err_idx = torch.as_tensor(noise_matrix < error_probability)

    corrupted_bitstrings = []
    for counter in counters:
        sample = sample_to_matrix(counter)
        corrupted_bitstrings.append(bs_corruption(err_idx=err_idx, sample=sample))
    return corrupted_bitstrings