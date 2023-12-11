from __future__ import annotations

from collections import Counter

import numpy as np


def shannon_entropy(counter: Counter) -> float:
    return float(-np.sum([count * np.log(count) for count in counter.values()]))


def js_divergence(counter_p: Counter, counter_q: Counter) -> float:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    represented as Counter objects.
    The JSD is calculated using only the shared keys between the two input Counter objects.

    Args:
        counter_p (Counter): Counter of bitstring counts for probability mass function P.
        counter_q (Counter): Counter of bitstring counts for probability mass function Q.

    Returns:
        float: The Jensen-Shannon divergence between counter_p and counter_q.
    """
    # Normalise counters
    normalisation_p = np.sum([count for count in counter_p.values()])
    normalisation_q = np.sum([count for count in counter_q.values()])
    counter_p = Counter({k: v / normalisation_p for k, v in counter_p.items()})
    counter_q = Counter({k: v / normalisation_q for k, v in counter_q.items()})

    average_proba_counter = counter_p + counter_q
    average_proba_counter = Counter({k: v / 2.0 for k, v in average_proba_counter.items()})
    average_entropy = shannon_entropy(average_proba_counter)

    entropy_p = shannon_entropy(counter_p)
    entropy_q = shannon_entropy(counter_q)
    return float(average_entropy - (entropy_p + entropy_q) / 2.0)


def norm_difference(counter_p: Counter, counter_q: Counter) -> float:
    # Normalise counters

    counter_p = np.array([v for v in counter_p.values()])
    counter_q = np.array([v for v in counter_q.values()])

    prob_p = counter_p / np.sum(counter_p)
    prob_q = counter_q / np.sum(counter_q)

    return float(np.linalg.norm(prob_p - prob_q))
