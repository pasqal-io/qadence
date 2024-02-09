from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from qadence.blocks import PrimitiveBlock
from qadence.parameters import evaluate
from torch import Tensor


def get_qubit_indices_for_op(pauli_term: tuple, op: PrimitiveBlock | None = None) -> list[int]:
    """Get qubit indices for the given op in the Pauli term if any."""
    indices = []
    blocks = getattr(pauli_term[0], "blocks", None)
    if blocks is not None:
        for block in blocks:
            if op is None:
                indices.append(block.qubit_support[0])
            if isinstance(block, type(op)):
                indices.append(block.qubit_support[0])
    else:
        block = pauli_term[0]
        if op is None:
            indices.append(block.qubit_support[0])
        if isinstance(block, type(op)):
            indices.append(block.qubit_support[0])
    return indices


def get_counts(samples: list, support: list) -> list:
    """Marginalise the probablity mass function to the support."""
    counts = []
    for sample in samples:
        sample_counts = []
        for k, v in sample.items():
            sample_counts.append(Counter({"".join([k[i] for i in support]): sample[k]}))
        reduced_counts = reduce(lambda x, y: x + y, sample_counts)
        counts.append(reduced_counts)
    return counts


def empirical_average(samples: list, support: list) -> Tensor:
    """Compute the empirical average."""
    counters = get_counts(samples, support)
    expectations = []
    n_shots = np.sum(list(counters[0].values()))
    parity = -1
    for counter in counters:
        counter_exps = []
        for bitstring, count in counter.items():
            counter_exps.append(count * parity ** (np.sum([int(bit) for bit in bitstring])))
        expectations.append(np.sum(counter_exps) / n_shots)
    return torch.tensor(expectations)


def iterate_pauli_decomposition(
    pauli_decomposition: list,
    samples: list,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms."""

    estimated_values = []
    for pauli_term in pauli_decomposition:
        support = get_qubit_indices_for_op(pauli_term)
        estim_values = empirical_average(samples=samples, support=support)
        # TODO: support for parametric observables to be tested
        estimated_values.append(estim_values * evaluate(pauli_term[1]))
    res = torch.sum(torch.stack(estimated_values), axis=0)
    return res
