from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import numpy.typing as npt
import torch
from scipy.linalg import norm
from scipy.optimize import LinearConstraint, minimize

from qadence.mitigations.protocols import Mitigations
from qadence.noise.protocols import Noise


def corrected_probas(p_corr: npt.NDArray, T: npt.NDArray, p_raw: npt.NDArray) -> np.double:
    return norm(T @ p_corr.T - p_raw.T, ord=2) ** 2


def renormalize_counts(corrected_counts: npt.NDArray, n_shots: int) -> npt.NDArray:
    """Renormalize counts rounding discrepancies."""
    total_counts = sum(corrected_counts)
    if total_counts != n_shots:
        counts_diff = total_counts - n_shots
        corrected_counts -= counts_diff
        corrected_counts = np.where(corrected_counts < 0, 0, corrected_counts)
        sum_corrected_counts = sum(corrected_counts)
        if sum_corrected_counts < n_shots:
            renormalization_factor = max(sum_corrected_counts, n_shots) / min(
                sum_corrected_counts, n_shots
            )
        else:
            renormalization_factor = min(sum_corrected_counts, n_shots) / max(
                sum_corrected_counts, n_shots
            )
        corrected_counts = np.rint(corrected_counts * renormalization_factor).astype(int)
    return corrected_counts


def mitigation_minimization(
    noise: Noise, mitigation: Mitigations, samples: list[Counter]
) -> list[Counter]:
    """Minimize a correction matrix subjected to stochasticity constraints.

    See Equation (5) in https://arxiv.org/pdf/2001.09980.pdf.
    """
    noise_matrices = noise.options.get("noise_matrix", noise.options["confusion_matrices"])
    n_qubits = len(list(samples[0].keys())[0])
    n_shots = sum(samples[0].values())
    # Build the whole T matrix.
    T_matrix = reduce(torch.kron, noise_matrices).detach().numpy()
    corrected_counters: list[Counter] = []
    for sample in samples:
        bitstring_length = 2**n_qubits
        # List of bitstrings in lexicographical order.
        ordered_bitstrings = [f"{i:0{n_qubits}b}" for i in range(bitstring_length)]
        # Array of raw probabilites.
        p_raw = np.array([sample[bs] for bs in ordered_bitstrings]) / n_shots
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
        res = minimize(corrected_probas, p_corr0, args=(T_matrix, p_raw), constraints=constraints)
        # breakpoint()
        corrected_counts = np.rint(res.x * n_shots).astype(int)
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


def mitigate(noise: Noise, mitigation: Mitigations, samples: list[Counter]) -> list[Counter]:
    return mitigation_minimization(noise=noise, mitigation=mitigation, samples=samples)
