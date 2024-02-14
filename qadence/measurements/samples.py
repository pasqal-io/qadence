from __future__ import annotations

from collections import Counter
from typing import List

from torch import Tensor

from qadence.blocks import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.measurements.utils import pauli_z_expectation


def compute_expectation(
    observables: List[AbstractBlock],
    samples: List[Counter],
) -> Tensor:
    """Given a list of observables in Z basis, compute the expectation values against samples.

    Args:
        observables: A list of observables
            to estimate the expectation values from.
        samples: List of samples against which expectation value is to be computed
    """

    if not isinstance(observables, list):
        raise TypeError(
            "Observables must be of type <class 'List[AbstractBlock]'>. Got type(observables)"
        )

    expectation_vals = []

    for observable in observables:
        decomposition = unroll_block_with_scaling(observable)
        if not (observable._is_diag_pauli and not observable.is_identity):
            raise TypeError("observable provided is not in the Z basis")
        expectation_vals.append(pauli_z_expectation(decomposition, samples))

    return expectation_vals
