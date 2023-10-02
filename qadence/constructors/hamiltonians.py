from __future__ import annotations

import warnings
from typing import List, Tuple, Type, Union

import numpy as np
import torch

from qadence.blocks import AbstractBlock, add
from qadence.logger import get_logger
from qadence.operations import N, X, Y, Z
from qadence.register import Register
from qadence.types import Interaction, TArray

logger = get_logger(__name__)


def interaction_zz(i: int, j: int) -> AbstractBlock:
    """Ising ZZ interaction."""
    return Z(i) @ Z(j)


def interaction_nn(i: int, j: int) -> AbstractBlock:
    """Ising NN interaction."""
    return N(i) @ N(j)


def interaction_xy(i: int, j: int) -> AbstractBlock:
    """XY interaction."""
    return X(i) @ X(j) + Y(i) @ Y(j)


def interaction_xyz(i: int, j: int) -> AbstractBlock:
    """Heisenberg XYZ interaction."""
    return X(i) @ X(j) + Y(i) @ Y(j) + Z(i) @ Z(j)


INTERACTION_DICT = {
    Interaction.ZZ: interaction_zz,
    Interaction.NN: interaction_nn,
    Interaction.XY: interaction_xy,
    Interaction.XYZ: interaction_xyz,
}


ARRAYS = (list, np.ndarray, torch.Tensor)

DETUNINGS = (N, X, Y, Z)

TDetuning = Union[Type[N], Type[X], Type[Y], Type[Z]]


def hamiltonian_factory(
    register: Register | int,
    interaction: Interaction | None = None,
    detuning: TDetuning | None = None,
    interaction_strength: TArray | str | None = None,
    detuning_strength: TArray | str | None = None,
    random_strength: bool = False,
    force_update: bool = False,
) -> AbstractBlock:
    """
    General Hamiltonian creation function. Can be used to create Hamiltonians with 2-qubit
    interactions and single-qubit detunings, both with arbitrary strength or parameterized.

    Arguments:
        register: register of qubits with a specific graph topology, or number of qubits.
            When passing a number of qubits a register with all-to-all connectivity
            is created.
        interaction: Interaction.ZZ, Interaction.NN, Interaction.XY, or Interacton.XYZ.
        detuning: single-qubit operator N, X, Y, or Z.
        interaction_strength: list of values to be used as the interaction strength for each
            pair of qubits. Should be ordered following the order of `Register(n_qubits).edges`.
            Alternatively, some string "x" can be passed, which will create a parameterized
            interactions for each pair of qubits, each labelled as `"x_ij"`.
        detuning_strength: list of values to be used as the detuning strength for each qubit.
            Alternatively, some string "x" can be passed, which will create a parameterized
            detuning for each qubit, each labelled as `"x_i"`.
        random_strength: set random interaction and detuning strengths between -1 and 1.
        force_update: force override register detuning and interaction strengths.

    Examples:
        ```python exec="on" source="material-block" result="json"
        from qadence import hamiltonian_factory, Interaction, Register, Z

        n_qubits = 3

        # Constant total magnetization observable:
        observable = hamiltonian_factory(n_qubits, detuning = Z)

        # Parameterized total magnetization observable:
        observable = hamiltonian_factory(n_qubits, detuning = Z, detuning_strength = "z")

        # Random all-to-all XY Hamiltonian generator:
        generator = hamiltonian_factory(
            n_qubits,
            interaction = Interaction.XY,
            random_strength = True,
            )

        # Parameterized NN Hamiltonian generator with a square grid interaction topology:
        register = Register.square(qubits_side = n_qubits)
        generator = hamiltonian_factory(
            register,
            interaction = Interaction.NN,
            interaction_strength = "theta"
            )
        ```
    """

    if interaction is None and detuning is None:
        raise ValueError("Please provide an interaction and/or detuning for the Hamiltonian.")

    # If number of qubits is given, creates all-to-all register
    register = Register(register) if isinstance(register, int) else register

    # Get interaction function
    try:
        int_fn = INTERACTION_DICT[interaction]  # type: ignore [index]
    except (KeyError, ValueError) as error:
        if interaction is None:
            pass
        else:
            raise KeyError(f"Interaction {interaction} not supported.")

    # Check single-qubit detuning
    if (detuning is not None) and (detuning not in DETUNINGS):
        raise TypeError(f"Detuning of type {type(detuning)} not supported.")

    # Pre-process detuning and interaction strengths and update register
    has_detuning_strength, detuning_strength = _preprocess_strengths(
        register, detuning_strength, "nodes", force_update, random_strength
    )
    has_interaction_strength, interaction_strength = _preprocess_strengths(
        register, interaction_strength, "edges", force_update, random_strength
    )

    if (not has_detuning_strength) or force_update:
        register = _update_detuning_strength(register, detuning_strength)

    if (not has_interaction_strength) or force_update:
        register = _update_interaction_strength(register, interaction_strength)

    # Create single-qubit detunings:
    single_qubit_terms: List[AbstractBlock] = []
    if detuning is not None:
        for node in register.nodes:
            block_sq = detuning(node)  # type: ignore [operator]
            strength_sq = register.nodes[node]["strength"]
            single_qubit_terms.append(strength_sq * block_sq)

    # Create two-qubit interactions:
    two_qubit_terms: List[AbstractBlock] = []
    if interaction is not None:
        for edge in register.edges:
            block_tq = int_fn(*edge)  # type: ignore [operator]
            strength_tq = register.edges[edge]["strength"]
            two_qubit_terms.append(strength_tq * block_tq)

    return add(*single_qubit_terms, *two_qubit_terms)


def _preprocess_strengths(
    register: Register,
    strength: TArray | str | None,
    nodes_or_edges: str,
    force_update: bool,
    random_strength: bool,
) -> Tuple[bool, Union[TArray | str]]:
    data = getattr(register, nodes_or_edges)

    # Useful for error messages:
    strength_target = "detuning" if nodes_or_edges == "nodes" else "interaction"

    # First we check if strength values already exist in the register
    has_strength = any(["strength" in data[i] for i in data])
    if has_strength and not force_update:
        if strength is not None:
            logger.warning(
                "Register already includes " + strength_target + " strengths. "
                "Skipping update. Use `force_update = True` to override them."
            )
    # Next we process the strength given in the input arguments
    if strength is None:
        if random_strength:
            strength = 2 * torch.rand(len(data), dtype=torch.double) - 1
        else:
            # None defaults to constant = 1.0
            strength = torch.ones(len(data), dtype=torch.double)
    elif isinstance(strength, ARRAYS):
        # If array is given, checks it has the correct length
        if len(strength) != len(data):
            message = "Array of " + strength_target + " strengths has incorrect size."
            raise ValueError(message)
    elif isinstance(strength, str):
        # Any string will be used as a prefix to variational parameters
        pass
    else:
        # If not of the accepted types ARRAYS or str, we error out
        raise TypeError(
            "Incorrect " + strength_target + f" strength type {type(strength)}. "
            "Please provide an array of strength values, or a string for "
            "parameterized " + strength_target + "s."
        )

    return has_strength, strength


def _update_detuning_strength(register: Register, detuning_strength: TArray | str) -> Register:
    for node in register.nodes:
        if isinstance(detuning_strength, str):
            register.nodes[node]["strength"] = detuning_strength + f"_{node}"
        elif isinstance(detuning_strength, ARRAYS):
            register.nodes[node]["strength"] = detuning_strength[node]
    return register


def _update_interaction_strength(
    register: Register, interaction_strength: TArray | str
) -> Register:
    for idx, edge in enumerate(register.edges):
        if isinstance(interaction_strength, str):
            register.edges[edge]["strength"] = interaction_strength + f"_{edge[0]}{edge[1]}"
        elif isinstance(interaction_strength, ARRAYS):
            register.edges[edge]["strength"] = interaction_strength[idx]
    return register


# FIXME: Previous hamiltonian / observable functions, now refactored, to be deprecated:

DEPRECATION_MESSAGE = "This function will be removed in the future. "


def single_z(qubit: int = 0, z_coefficient: float = 1.0) -> AbstractBlock:
    message = DEPRECATION_MESSAGE + "Please use `z_coefficient * Z(qubit)` directly."
    warnings.warn(message, FutureWarning)
    return Z(qubit) * z_coefficient


def total_magnetization(n_qubits: int, z_terms: np.ndarray | list | None = None) -> AbstractBlock:
    message = (
        DEPRECATION_MESSAGE
        + "Please use `hamiltonian_factory(n_qubits, detuning=Z, node_coeff=z_terms)`."
    )
    warnings.warn(message, FutureWarning)
    return hamiltonian_factory(n_qubits, detuning=Z, detuning_strength=z_terms)


def zz_hamiltonian(
    n_qubits: int,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AbstractBlock:
    message = (
        DEPRECATION_MESSAGE
        + """
Please use `hamiltonian_factory(n_qubits, Interaction.ZZ, Z, interaction_strength, z_terms)`. \
Note that the argument `zz_terms` in this function is a 2D array of size `(n_qubits, n_qubits)`, \
while `interaction_strength` is expected as a 1D array of size `0.5 * n_qubits * (n_qubits - 1)`."""
    )
    warnings.warn(message, FutureWarning)
    if zz_terms is not None:
        register = Register(n_qubits)
        interaction_strength = [zz_terms[edge[0], edge[1]] for edge in register.edges]
    else:
        interaction_strength = None

    return hamiltonian_factory(n_qubits, Interaction.ZZ, Z, interaction_strength, z_terms)


def ising_hamiltonian(
    n_qubits: int,
    x_terms: np.ndarray | None = None,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AbstractBlock:
    message = (
        DEPRECATION_MESSAGE
        + """
You can build a general transverse field ising model with the `hamiltonian_factory` function. \
Check the hamiltonian construction tutorial in the documentation for more information."""
    )
    warnings.warn(message, FutureWarning)
    zz_ham = zz_hamiltonian(n_qubits, z_terms=z_terms, zz_terms=zz_terms)
    x_ham = hamiltonian_factory(n_qubits, detuning=X, detuning_strength=x_terms)
    return zz_ham + x_ham
