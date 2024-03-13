from __future__ import annotations

from collections import Counter
from enum import Enum
from logging import getLogger

import torch
from torch import Tensor
from torch.distributions import normal, poisson, uniform

logger = getLogger(__name__)


class WhiteNoise(Enum):
    """White noise distributions."""

    UNIFORM = staticmethod(uniform.Uniform(low=0.0, high=1.0))
    """Uniform white noise."""

    GAUSSIAN = staticmethod(normal.Normal(loc=0.0, scale=1.0))
    """Gaussian white noise."""

    POISSON = staticmethod(poisson.Poisson(rate=0.1))
    """Poisson white noise."""


def bitstring_to_tensor(bitstring: str) -> Tensor:
    """
    A helper function to convert bit strings to torch.Tensor.

    Args:
        bitstring:  A str format of a bit string.

    Returns:
        A torch.Tensor out of the input bit string.
    """
    return torch.as_tensor(list(map(int, bitstring)))


def tensor_to_bitstring(bitstring: Tensor) -> str:
    """
    A helper function to convert torch.Tensor to bit strings.

    Args:
        bitstring: A torch.Tensor format of a bit string.

    Returns:
        A str out of the input bit string.
    """
    return "".join(list(map(str, bitstring.detach().tolist())))


def bit_flip(bit: Tensor, cond: Tensor) -> Tensor:
    """
    A helper function that reverses the states 0 and 1 in the bit string.

    Args:
        bit: A integer-value bit in a bitstring to be inverted.
        cond: A Bool value of whether or not a bit should be flipped.

    Returns:
        The inverse value of the input bit
    """
    return torch.where(
        cond,
        torch.where(
            bit == torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        ),
        bit,
    )


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
        Uniform, or Poisson.
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


def create_confusion_matrices(noise_matrix: Tensor, error_probability: float) -> Tensor:
    confusion_matrices = []
    for i in range(noise_matrix.size()[1]):
        column_tensor = noise_matrix[:, i]
        flip_proba = column_tensor[column_tensor < error_probability].mean().item()
        confusion_matrix = torch.tensor(
            [[1.0 - flip_proba, flip_proba], [flip_proba, 1.0 - flip_proba]], dtype=torch.float64
        )
        confusion_matrices.append(confusion_matrix)
    return torch.stack(confusion_matrices)


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
    error_probability = options.get("error_probability")
    noise_distribution = options.get("noise_distribution", WhiteNoise.UNIFORM)
    noise_matrix = options.get("noise_matrix")

    # option for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    if error_probability is None:
        error_probability = 0.1
        # Return the default error probability for mitigation purposes.
        options["default_error_probability"] = error_probability
    if noise_matrix is None:
        # assumes that all bits can be flipped independently of each other
        noise_matrix = create_noise_matrix(noise_distribution, n_shots, n_qubits)
        confusion_matrices = create_confusion_matrices(
            noise_matrix=noise_matrix, error_probability=error_probability
        )
        # Return the generated noise matrix for mitigation purposes.
        options["confusion_matrices"] = confusion_matrices
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
