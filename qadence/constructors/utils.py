from __future__ import annotations

from typing import Iterable, Type

import numpy as np
import sympy

from qadence.blocks import KronBlock, kron
from qadence.operations import RY
from qadence.parameters import FeatureParameter, Parameter


def generator_prefactor(spectrum: str, qubit_index: int) -> float | int:
    """Converts a spectrum string, e.g. tower or exponential.

    The result is the correct generator prefactor.
    """
    spectrum = spectrum.lower()
    conversion_dict: dict[str, float | int] = {
        "simple": 1,
        "tower": qubit_index + 1,
        "exponential": 2 * np.pi / (2 ** (qubit_index + 1)),
    }
    return conversion_dict[spectrum]


def basis_func(basis: str, x: Parameter) -> Parameter | sympy.Expr:
    basis = basis.lower()
    conversion_dict: dict[str, Parameter | sympy.Expr] = {
        "fourier": x,
        "chebyshev": 2 * sympy.acos(x),
    }
    return conversion_dict[basis]


def build_idx_fms(
    basis: str,
    fm_pauli: Type[RY],
    fm_strategy: str,
    n_features: int,
    n_qubits: int,
    spectrum: str,
) -> list[KronBlock]:
    """Builds the index feature maps based on the given parameters.

    Args:
        basis (str): Type of basis chosen for the feature map.
        fm_pauli (PrimitiveBlock type): The chosen Pauli rotation type.
        fm_strategy (str): The feature map strategy to be used. Possible values are
            'parallel' or 'serial'.
        n_features (int): The number of features.
        n_qubits (int): The number of qubits.
        spectrum (str): The chosen spectrum.

    Returns:
        List[KronBlock]: The list of index feature maps.
    """
    idx_fms = []
    for i in range(n_features):
        target_qubits = get_fm_qubits(fm_strategy, i, n_qubits, n_features)
        param = FeatureParameter(f"x{i}")
        block = kron(
            *[
                fm_pauli(qubit, generator_prefactor(spectrum, j) * basis_func(basis, param))
                for j, qubit in enumerate(target_qubits)
            ]
        )
        idx_fm = block
        idx_fms.append(idx_fm)
    return idx_fms


def get_fm_qubits(fm_strategy: str, i: int, n_qubits: int, n_features: int) -> Iterable:
    """Returns the list of target qubits for the given feature map strategy and feature index.

    Args:
        fm_strategy (str): The feature map strategy to be used. Possible values
            are 'parallel' or 'serial'.
        i (int): The feature index.
        n_qubits (int): The number of qubits.
        n_features (int): The number of features.

    Returns:
        List[int]: The list of target qubits.

    Raises:
        ValueError: If the feature map strategy is not implemented.
    """
    if fm_strategy == "parallel":
        n_qubits_per_feature = int(n_qubits / n_features)
        target_qubits = range(i * n_qubits_per_feature, (i + 1) * n_qubits_per_feature)
    elif fm_strategy == "serial":
        target_qubits = range(0, n_qubits)
    else:
        raise ValueError(f"Feature map strategy {fm_strategy} not implemented.")
    return target_qubits
