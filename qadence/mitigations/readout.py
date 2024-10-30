from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import numpy.typing as npt
import torch
from numpy.linalg import inv, matrix_rank, pinv
from scipy.linalg import norm
from scipy.optimize import LinearConstraint, minimize

from qadence.mitigations.protocols import Mitigations
from qadence.noise.protocols import NoiseHandler
from qadence.types import NoiseProtocol, ReadOutOptimization


def corrected_probas(p_corr: npt.NDArray, T: npt.NDArray, p_raw: npt.NDArray) -> np.double:
    return norm(T @ p_corr.T - p_raw.T, ord=2) ** 2


def mle_solve(p_raw: npt.NDArray) -> npt.NDArray:
    """
    Compute the MLE probability vector.

    Algorithmic details can be found in https://arxiv.org/pdf/1106.5458.pdf Page(3).
    """
    # Sort p_raw by values while keeping track of indices.
    index_sort = p_raw.argsort()
    p_sort = p_raw[index_sort]
    neg_sum = 0
    breakpoint = len(p_sort) - 1

    for i in range(len(p_sort)):
        ## if neg_sum cannot be distributed among other probabilities, continue to accumulate
        if p_sort[i] + neg_sum / (len(p_sort) - i) < 0:
            neg_sum += p_sort[i]
            p_sort[i] = 0
        # set breakpoint to current index
        else:
            breakpoint = i
            break
    ## number of entries to which i can distribute(includes breakpoint)
    size = len(p_sort) - breakpoint
    p_sort[breakpoint:] += neg_sum / size

    re_index_sort = index_sort.argsort()
    p_corr = p_sort[re_index_sort]

    return p_corr


def renormalize_counts(corrected_counts: npt.NDArray, n_shots: int) -> npt.NDArray:
    """Renormalize counts rounding discrepancies."""
    total_counts = sum(corrected_counts)
    if total_counts != n_shots:
        counts_diff = total_counts - n_shots
        corrected_counts -= counts_diff
        corrected_counts = np.where(corrected_counts < 0, 0, corrected_counts)
        sum_corrected_counts = sum(corrected_counts)

        renormalization_factor = n_shots / sum_corrected_counts
        corrected_counts = np.rint(corrected_counts * renormalization_factor).astype(int)
    return corrected_counts


def matrix_inv(K: npt.NDArray) -> npt.NDArray:
    return inv(K) if matrix_rank(K) == K.shape[0] else pinv(K)


def mitigation_minimization(
    noise: NoiseHandler,
    mitigation: Mitigations,
    samples: list[Counter],
) -> list[Counter]:
    """Minimize a correction matrix subjected to stochasticity constraints.

    See Equation (5) in https://arxiv.org/pdf/2001.09980.pdf.
    See Page(3) in https://arxiv.org/pdf/1106.5458.pdf for MLE implementation

    Args:
        noise: Specifies confusion matrix and default error probability
        mitigation: Selects additional mitigation options based on noise choice.
        For readout we have the following mitigation options for optimization
        1. constrained 2. mle. Default : mle
        samples: List of samples to be mitigated

    Returns:
        Mitigated counts computed by the algorithm
    """
    protocol, options = noise.protocol[-1], noise.options[-1]
    if protocol != NoiseProtocol.READOUT:
        raise ValueError("Specify a noise source of type NoiseProtocol.READOUT.")
    noise_matrices = options.get("noise_matrix", options["confusion_matrices"])
    optimization_type = mitigation.options.get("optimization_type", ReadOutOptimization.MLE)
    n_qubits = len(list(samples[0].keys())[0])
    n_shots = sum(samples[0].values())
    corrected_counters: list[Counter] = []

    if optimization_type == ReadOutOptimization.CONSTRAINED:
        # Build the whole T matrix.
        T_matrix = reduce(torch.kron, noise_matrices).detach().numpy()

    if optimization_type == ReadOutOptimization.MLE:
        # Check if matrix is singular and use appropriate inverse.
        noise_matrices_inv = list(map(matrix_inv, noise_matrices.numpy()))
        T_inv = reduce(np.kron, noise_matrices_inv)

    for sample in samples:
        bitstring_length = 2**n_qubits
        # List of bitstrings in lexicographical order.
        ordered_bitstrings = [f"{i:0{n_qubits}b}" for i in range(bitstring_length)]
        # Array of raw probabilites.
        p_raw = np.array([sample[bs] for bs in ordered_bitstrings]) / n_shots

        if optimization_type == ReadOutOptimization.CONSTRAINED:
            # Initial random guess in [0,1].
            p_corr0 = np.random.rand(bitstring_length)
            # Stochasticity constraints.
            normality_constraint = LinearConstraint(
                np.ones(bitstring_length).astype(int), lb=1.0, ub=1.0
            )
            positivity_constraint = LinearConstraint(
                np.eye(bitstring_length).astype(int), lb=0.0, ub=1.0
            )
            constraints = [normality_constraint, positivity_constraint]
            # Minimize the corrected probabilities.
            res = minimize(
                corrected_probas, p_corr0, args=(T_matrix, p_raw), constraints=constraints
            )
            p_corr = res.x

        elif optimization_type == ReadOutOptimization.MLE:
            # Compute corrected inverse using matrix inversion and run MLE.
            p_corr = mle_solve(T_inv @ p_raw)
        else:
            raise NotImplementedError(
                f"Requested method {optimization_type} does not match supported protocols."
            )

        corrected_counts = np.rint(p_corr * n_shots).astype(int)

        # Renormalize if total counts differs from n_shots.
        corrected_counts = renormalize_counts(corrected_counts=corrected_counts, n_shots=n_shots)
        # At this point, the count should be off by at most 2, added or substracted to/from the
        # max count.
        if sum(corrected_counts) != n_shots:
            count_diff = sum(corrected_counts) - n_shots
            max_count_bs = np.argmax(corrected_counts)
            corrected_counts[max_count_bs] -= count_diff
        assert (
            corrected_counts.sum() == n_shots
        ), f"Corrected counts sum: {corrected_counts.sum()}, n_shots: {n_shots}"
        corrected_counters.append(
            Counter(
                {bs: count for bs, count in zip(ordered_bitstrings, corrected_counts) if count > 0}
            )
        )
    return corrected_counters


def mitigate(noise: NoiseHandler, mitigation: Mitigations, samples: list[Counter]) -> list[Counter]:
    return mitigation_minimization(noise=noise, mitigation=mitigation, samples=samples)
