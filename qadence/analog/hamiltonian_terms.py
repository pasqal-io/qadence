from __future__ import annotations

from torch import tensor

from qadence.analog.device import RydbergDevice
from qadence.analog.utils import C6_DICT
from qadence.blocks.abstract import AbstractBlock
from qadence.constructors import hamiltonian_factory
from qadence.register import Register
from qadence.types import Interaction


def rydberg_interaction_hamiltonian(
    register: Register, device_specs: RydbergDevice
) -> AbstractBlock:
    """
    Computes the Rydberg Ising or XY interaction Hamiltonian for a register of qubits.

    H_int = ∑_(j<i) (C_6 / R**6) * kron(N_i, N_j)

    H_int = ∑_(j<i) (C_3 / R**3) * (kron(X_i, X_j) + kron(Y_i, Y_j))

    Args:
        register: the register of qubits.
        device: the RydbergDevice with respective specs.
    """

    if device_specs.interaction == Interaction.NN:
        c6 = C6_DICT[device_specs.rydberg_level]
        strength_list = c6 / (tensor(register.distances) ** 6)
    elif device_specs.interaction == Interaction.XY:
        strength_list = device_specs.coeff_xy / (tensor(register.distances) ** 3)

    return hamiltonian_factory(
        register,
        interaction=device_specs.interaction,
        interaction_strength=strength_list,
        use_complete_graph=True,
    )
