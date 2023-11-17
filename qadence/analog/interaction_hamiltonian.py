from __future__ import annotations

from math import dist as euclidean_distance

from torch import Tensor, float64, tensor

from qadence.analog.device import RydbergDevice
from qadence.analog.utils import C6_DICT
from qadence.blocks.abstract import AbstractBlock
from qadence.constructors import hamiltonian_factory
from qadence.register import Register
from qadence.types import Interaction


def _distance_all_qubits(r: Register) -> Tensor:
    return tensor(
        [euclidean_distance(r.coords[e[0]], r.coords[e[1]]) for e in r.all_edges], dtype=float64
    )


def _nn_strength(register: Register, rydberg_level: int) -> Tensor:
    """(C_6 / R_ij**6)."""
    c6 = C6_DICT[rydberg_level]
    return c6 / (_distance_all_qubits(register) ** 6)


def _xy_strength(register: Register, coeff_xy: float) -> Tensor:
    """(C_3 / R_ij**3)."""
    return coeff_xy / (_distance_all_qubits(register) ** 3)


def rydberg_interaction_hamiltonian(register: Register, device: RydbergDevice) -> AbstractBlock:
    """
    Computes the Rydberg Ising or XY interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_6 / R**6) * kron(N_i, N_j)

    H_int = ∑_(j<i) (C_3 / R**3) * (kron(X_i, X_j) + kron(Y_i, Y_j))

    Args:
        register: the register of qubits.
        device: the RydbergDevice with respective specs.
    """

    if device.interaction == Interaction.NN:
        strength_list = _nn_strength(register, device.rydberg_level)
    elif device.interaction == Interaction.XY:
        strength_list = _xy_strength(register, device.coeff_xy)

    return hamiltonian_factory(
        register,
        interaction=device.interaction,
        interaction_strength=strength_list,
        use_complete_graph=True,
    )
