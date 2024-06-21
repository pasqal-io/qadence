from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Callable, List, Type, Union

import numpy as np
from torch import Tensor, double, ones, rand
from typing_extensions import Any

from qadence.blocks import AbstractBlock, add, block_is_qubit_hamiltonian
from qadence.operations import N, X, Y, Z
from qadence.register import Register
from qadence.types import Interaction, ObservableTransform, TArray, TParameter

logger = getLogger(__name__)


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


ARRAYS = (list, np.ndarray, Tensor)

DETUNINGS = (N, X, Y, Z)

TDetuning = Union[Type[N], Type[X], Type[Y], Type[Z]]


def hamiltonian_factory(
    register: Register | int,
    interaction: Interaction | Callable | None = None,
    detuning: TDetuning | None = None,
    interaction_strength: TArray | str | None = None,
    detuning_strength: TArray | str | None = None,
    random_strength: bool = False,
    use_all_node_pairs: bool = False,
) -> AbstractBlock:
    """
    General Hamiltonian creation function.

    Can be used to create Hamiltonians with 2-qubit
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
        use_all_node_pairs: computes an interaction term for every pair of nodes in the graph,
            independent of the edge topology in the register. Useful for defining Hamiltonians
            where the interaction strength decays with the distance.

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
    if interaction is not None:
        if callable(interaction):
            int_fn = interaction
            try:
                if not block_is_qubit_hamiltonian(interaction(0, 1)):
                    raise ValueError("Custom interactions must be composed of Pauli operators.")
            except TypeError:
                raise TypeError(
                    "Please use a custom interaction function signed with two integer parameters."
                )
        else:
            int_fn = INTERACTION_DICT.get(interaction, None)  # type: ignore [arg-type]
            if int_fn is None:
                raise KeyError(f"Interaction {interaction} not supported.")

    # Check single-qubit detuning
    if (detuning is not None) and (detuning not in DETUNINGS):
        raise TypeError(f"Detuning of type {type(detuning)} not supported.")

    # Pre-process detuning and interaction strengths and update register
    detuning_strength_array = _preprocess_strengths(
        register, detuning_strength, "nodes", random_strength
    )

    edge_str = "all_node_pairs" if use_all_node_pairs else "edges"
    interaction_strength_array = _preprocess_strengths(
        register, interaction_strength, edge_str, random_strength
    )

    # Create single-qubit detunings:
    single_qubit_terms: List[AbstractBlock] = []
    if detuning is not None:
        for strength, node in zip(detuning_strength_array, register.nodes):
            single_qubit_terms.append(strength * detuning(node))

    # Create two-qubit interactions:
    two_qubit_terms: List[AbstractBlock] = []
    edge_data = register.all_node_pairs if use_all_node_pairs else register.edges
    if interaction is not None and int_fn is not None:
        for strength, edge in zip(interaction_strength_array, edge_data):
            two_qubit_terms.append(strength * int_fn(*edge))

    return add(*single_qubit_terms, *two_qubit_terms)


def _preprocess_strengths(
    register: Register,
    strength: TArray | str | None,
    nodes_or_edges: str,
    random_strength: bool,
) -> Tensor | list:
    data = getattr(register, nodes_or_edges)

    # Useful for error messages:
    strength_target = "detuning" if nodes_or_edges == "nodes" else "interaction"

    # Next we process the strength given in the input arguments
    if strength is None:
        if random_strength:
            strength = 2 * rand(len(data), dtype=double) - 1
        else:
            # None defaults to constant = 1.0
            strength = ones(len(data), dtype=double)
    elif isinstance(strength, ARRAYS):
        # If array is given, checks it has the correct length
        if len(strength) != len(data):
            message = "Array of " + strength_target + " strengths has incorrect size."
            raise ValueError(message)
    elif isinstance(strength, str):
        prefix = strength
        if nodes_or_edges == "nodes":
            strength = [prefix + f"_{node}" for node in data]
        if nodes_or_edges in ["edges", "all_node_pairs"]:
            strength = [prefix + f"_{edge[0]}{edge[1]}" for edge in data]
    else:
        # If not of the accepted types ARRAYS or str, we error out
        raise TypeError(
            "Incorrect " + strength_target + f" strength type {type(strength)}. "
            "Please provide an array of strength values, or a string for "
            "parameterized " + strength_target + "s."
        )

    return strength


def total_magnetization(n_qubits: int, z_terms: np.ndarray | list | None = None) -> AbstractBlock:
    return hamiltonian_factory(n_qubits, detuning=Z, detuning_strength=z_terms)


def zz_hamiltonian(
    n_qubits: int,
    z_terms: np.ndarray | None = None,
    zz_terms: np.ndarray | None = None,
) -> AbstractBlock:
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
    zz_ham = zz_hamiltonian(n_qubits, z_terms=z_terms, zz_terms=zz_terms)
    x_ham = hamiltonian_factory(n_qubits, detuning=X, detuning_strength=x_terms)
    return zz_ham + x_ham


def is_numeric(x: Any) -> bool:
    return type(x) in (int, float, complex, np.int64, np.float64)


@dataclass
class ObservableConfig:
    detuning: TDetuning
    """
    Single qubit detuning of the observable Hamiltonian.

    Accepts single-qubit operator N, X, Y, or Z.
    """
    scale: TParameter = 1.0
    """The scale by which to multiply the output of the observable."""
    shift: TParameter = 0.0
    """The shift to add to the output of the observable."""
    transformation_type: ObservableTransform = ObservableTransform.NONE  # type: ignore[assignment]
    """The type of transformation."""
    trainable_transform: bool | None = None
    """
    Whether to have a trainable transformation on the output of the observable.

    If None, the scale and shift are numbers.
    If True, the scale and shift are VariationalParameter.
    If False, the scale and shift are FeatureParameter.
    """

    def __post_init__(self) -> None:
        if is_numeric(self.scale) and is_numeric(self.shift):
            assert (
                self.trainable_transform is None
            ), f"If scale and shift are numbers, trainable_transform must be None. \
            But got: {self.trainable_transform}"
