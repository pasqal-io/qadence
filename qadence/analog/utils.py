from __future__ import annotations

from math import dist as euclidean_distance

from sympy import cos, sin

from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.analog import (
    ConstantAnalogRotation,
)
from qadence.blocks.utils import add, kron
from qadence.operations import N, X, Y
from qadence.register import Register

# Ising coupling coefficient depending on the Rydberg level
# Include a normalization to the Planck constant hbar
# In units of [rad . µm^6 / µs]

C6_DICT = {
    50: 96120.72,
    51: 122241.6,
    52: 154693.02,
    53: 194740.36,
    54: 243973.91,
    55: 304495.01,
    56: 378305.98,
    57: 468027.05,
    58: 576714.85,
    59: 707911.38,
    60: 865723.02,
    61: 1054903.11,
    62: 1281042.11,
    63: 1550531.15,
    64: 1870621.31,
    65: 2249728.57,
    66: 2697498.69,
    67: 3224987.51,
    68: 3844734.37,
    69: 4571053.32,
    70: 5420158.53,
    71: 6410399.4,
    72: 7562637.31,
    73: 8900342.14,
    74: 10449989.62,
    75: 12241414.53,
    76: 14308028.03,
    77: 16687329.94,
    78: 19421333.62,
    79: 22557029.94,
    80: 26146720.74,
    81: 30248886.65,
    82: 34928448.69,
    83: 40257623.67,
    84: 46316557.88,
    85: 53194043.52,
    86: 60988354.64,
    87: 69808179.15,
    88: 79773468.88,
    89: 91016513.07,
    90: 103677784.57,
    91: 117933293.96,
    92: 133943541.9,
    93: 151907135.94,
    94: 172036137.34,
    95: 194562889.89,
    96: 219741590.56,
    97: 247850178.91,
    98: 279192193.77,
    99: 314098829.39,
    100: 352931119.11,
}


def _qubitposition(register: Register, i: int) -> tuple[int, int]:
    (x, y) = list(register.coords.values())[i]
    return (x, y)


def ising_interaction(
    register: Register, pairs: list[tuple[int, int]], rydberg_level: int = 60
) -> AbstractBlock:
    """
    Computes the Rydberg Ising interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_6 / R**6) * kron(N_i, N_j)

    Args:
        register: the register of qubits.
        pairs: a list of all pairs of interacting qubits.
        rydberg_level: determines the value of C_6
    """
    c6 = C6_DICT[rydberg_level]

    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c6 / rij**6) * kron(N(i), N(j))

    return add(term(i, j) for (i, j) in pairs)


def xy_interaction(
    register: Register, pairs: list[tuple[int, int]], c3: float = 3700.0
) -> AbstractBlock:
    """
    Computes the Rydberg XY interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_3 / R**3) * (kron(X_i, X_j) + kron(Y_i, Y_j))

    Args:
        register: the register of qubits.
        pairs: a list of all pairs of interacting qubits.
        c3: the coefficient value of C_3 in units of [rad . µm^3 / µs]
    """

    def term(i: int, j: int) -> AbstractBlock:
        qi, qj = _qubitposition(register, i), _qubitposition(register, j)
        rij = euclidean_distance(qi, qj)
        return (c3 / rij**3) * (kron(X(i), X(j)) + kron(Y(i), Y(j)))

    return add(term(i, j) for (i, j) in pairs)


def rot_generator(block: ConstantAnalogRotation) -> AbstractBlock:
    omega = block.parameters.omega
    delta = block.parameters.delta
    phase = block.parameters.phase
    support = block.qubit_support

    x_terms = (omega / 2) * add(cos(phase) * X(i) - sin(phase) * Y(i) for i in support)
    z_terms = delta * add(N(i) for i in support)
    return x_terms - z_terms  # type: ignore[no-any-return]
