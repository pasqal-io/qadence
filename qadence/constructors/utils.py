from __future__ import annotations

from typing import Iterable, Type

import sympy

from qadence.blocks import KronBlock, kron
from qadence.operations import RY
from qadence.parameters import FeatureParameter, Parameter
from qadence.types import PI, BasisSet, MultivariateStrategy, ReuploadScaling


def generator_prefactor(reupload_scaling: ReuploadScaling, qubit_index: int) -> float | int:
    """Converts a spectrum string, e.g. tower or exponential.

    The result is the correct generator prefactor.
    """
    conversion_dict: dict[str, float | int] = {
        ReuploadScaling.CONSTANT: 1,
        ReuploadScaling.TOWER: qubit_index + 1,
        ReuploadScaling.EXP: 2 * PI / (2 ** (qubit_index + 1)),
    }
    return conversion_dict[reupload_scaling]


def basis_func(basis: BasisSet, x: Parameter) -> Parameter | sympy.Expr:
    conversion_dict: dict[str, Parameter | sympy.Expr] = {
        BasisSet.FOURIER: x,
        BasisSet.CHEBYSHEV: 2 * sympy.acos(x),
    }
    return conversion_dict[basis]


def build_idx_fms(
    basis: BasisSet,
    fm_pauli: Type[RY],
    multivariate_strategy: MultivariateStrategy,
    n_features: int,
    n_qubits: int,
    reupload_scaling: ReuploadScaling,
) -> list[KronBlock]:
    """Builds the index feature maps based on the given parameters.

    Args:
        basis (BasisSet): Type of basis chosen for the feature map.
        fm_pauli (PrimitiveBlock type): The chosen Pauli rotation type.
        multivariate_strategy (MultivariateStrategy): The strategy used for encoding
            the multivariate feature map.
        n_features (int): The number of features.
        n_qubits (int): The number of qubits.
        reupload_scaling (ReuploadScaling): The chosen scaling for the reupload.

    Returns:
        List[KronBlock]: The list of index feature maps.
    """
    idx_fms = []
    for i in range(n_features):
        target_qubits = get_fm_qubits(multivariate_strategy, i, n_qubits, n_features)
        param = FeatureParameter(f"x{i}")
        block = kron(
            *[
                fm_pauli(qubit, generator_prefactor(reupload_scaling, j) * basis_func(basis, param))
                for j, qubit in enumerate(target_qubits)
            ]
        )
        idx_fm = block
        idx_fms.append(idx_fm)
    return idx_fms


def get_fm_qubits(
    multivariate_strategy: MultivariateStrategy, i: int, n_qubits: int, n_features: int
) -> Iterable:
    """Returns the list of target qubits for the given feature map strategy and feature index.

    Args:
        multivariate_strategy (MultivariateStrategy): The strategy used for encoding
            the multivariate feature map.
        i (int): The feature index.
        n_qubits (int): The number of qubits.
        n_features (int): The number of features.

    Returns:
        List[int]: The list of target qubits.

    Raises:
        ValueError: If the feature map strategy is not implemented.
    """
    if multivariate_strategy == MultivariateStrategy.PARALLEL:
        n_qubits_per_feature = int(n_qubits / n_features)
        target_qubits = range(i * n_qubits_per_feature, (i + 1) * n_qubits_per_feature)
    elif multivariate_strategy == MultivariateStrategy.SERIES:
        target_qubits = range(0, n_qubits)
    else:
        raise ValueError(f"Multivariate strategy {multivariate_strategy} not implemented.")
    return target_qubits
