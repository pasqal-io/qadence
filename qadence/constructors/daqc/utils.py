from __future__ import annotations

from logging import getLogger

import torch

logger = getLogger(__name__)


def _k_d(a: int, b: int) -> int:
    """Kronecker delta."""
    return int(a == b)


def _ix_map(n: int, a: int, b: int) -> int:
    """Maps `(a, b)` with `b` in [1, n] and `a < b` to range [1, n(n-1)/2]."""
    return int(n * (a - 1) - 0.5 * a * (a + 1) + b)


def _build_matrix_M(n_qubits: int) -> torch.Tensor:
    """Sign matrix used by the DAQC technique for the Ising model."""
    flat_size = int(0.5 * n_qubits * (n_qubits - 1))

    def matrix_M_ix(j: int, k: int, n: int, m: int) -> float:
        return (-1.0) ** (_k_d(n, j) + _k_d(n, k) + _k_d(m, j) + _k_d(m, k))

    M = torch.zeros(flat_size, flat_size)
    for k in range(2, n_qubits + 1):
        for j in range(1, k):
            for m in range(2, n_qubits + 1):
                for n in range(1, m):
                    alpha = _ix_map(n_qubits, n, m)
                    beta = _ix_map(n_qubits, j, k)
                    M[alpha - 1, beta - 1] = matrix_M_ix(j, k, n, m)
    return M
