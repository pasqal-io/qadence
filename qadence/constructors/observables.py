from __future__ import annotations

import numpy as np

from qadence.blocks import AbstractBlock, AddBlock, add, kron
from qadence.operations import N, X, Z


def single_z(qubit: int = 0, z_coefficient: float = 1.0) -> AbstractBlock:
    return Z(qubit) * z_coefficient


def _total_magnetization(n_qubits: int, z_terms: np.ndarray | list | None = None) -> AddBlock:
    coefficients = z_terms if z_terms is not None else [1.0] * n_qubits
    if (
        not (isinstance(coefficients, np.ndarray) or isinstance(coefficients, list))
        or len(coefficients) != n_qubits
    ):
        raise TypeError(f"z_terms should be a list or np.ndarray of length {n_qubits}")

    return add(Z(i) * c for (i, c) in enumerate(coefficients))


def total_magnetization(n_qubits: int, z_terms: np.ndarray | list | None = None) -> AbstractBlock:
    return _total_magnetization(n_qubits, z_terms)


def _zz_hamiltonian(
    n_qubits: int,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AddBlock:
    hamiltonian = _total_magnetization(n_qubits, z_terms=z_terms)

    zz_coefficients = zz_terms if zz_terms is not None else np.ones((n_qubits, n_qubits))
    if not isinstance(zz_coefficients, np.ndarray) or zz_coefficients.shape[0] != n_qubits:
        raise TypeError(f"z_array should be a list or np.ndarray of length {n_qubits}")

    zz = []
    for qubit in range(n_qubits):
        for qubit2 in range(qubit + 1, n_qubits):
            b = kron(Z(qubit), Z(qubit2)) * zz_coefficients[qubit, qubit2]
            zz.append(b)

    return add(hamiltonian, *zz)


def zz_hamiltonian(
    n_qubits: int,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AbstractBlock:
    return _zz_hamiltonian(n_qubits, z_terms, zz_terms)


def ising_hamiltonian(
    n_qubits: int,
    x_terms: np.ndarray | None = None,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AbstractBlock:
    hamiltonian = _zz_hamiltonian(n_qubits, z_terms=z_terms, zz_terms=zz_terms)

    x_coefficients = x_terms if x_terms is not None else np.ones(n_qubits)
    if not isinstance(x_coefficients, np.ndarray) or x_coefficients.shape[0] != n_qubits:
        raise TypeError(f"z_array should be a list or np.ndarray of length {n_qubits}")

    return hamiltonian + add(X(i) * c for (i, c) in enumerate(x_coefficients))


def nn_hamiltonian(n_qubits: int) -> AbstractBlock:
    """To be refactored"""
    terms = []
    for j in range(n_qubits):
        for i in range(j):
            terms.append(N(i) @ N(j))
    return add(*terms)
