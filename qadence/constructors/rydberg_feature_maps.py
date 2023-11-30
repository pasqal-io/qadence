from __future__ import annotations

from typing import Callable

import numpy as np
from sympy import Basic, Function

from qadence import (
    AbstractBlock,
    AnalogRot,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    FeatureParameter,
    Parameter,
    kron,
)
from qadence.blocks import AnalogBlock
from qadence.constructors.feature_maps import fm_parameter
from qadence.logger import get_logger
from qadence.types import BasisSet, TParameter

logger = get_logger(__file__)

AnalogRotationTypes = [AnalogRX, AnalogRY, AnalogRZ]


def rydberg_tower_feature_map(
    n_qubits: int,
    param: str = "phi",
    max_abs_detuning: float = 2 * np.pi * 10,
    tower_weights: list[float] | None = None,
) -> AbstractBlock:
    """Feature map using semi-local addressing patterns.

    Args:
        n_qubits (int): number of qubits
        param: the name of the feature parameter
        max_abs_detuning: maximum value of absolute detuning for each qubit
        tower_weights: a list of wegiths to assign to each qubit in the tower feature map

    Returns:
        AbstractBlock: _description_
    """
    max_abs_detuning = 2 * np.pi * 10
    tower_coeffs = list(np.arange(1, n_qubits + 1)) if tower_weights is None else tower_weights
    tower_detuning = max_abs_detuning / (sum(tower_coeffs[i] for i in range(n_qubits)))

    param = FeatureParameter(param)
    duration = 1000 * param / tower_detuning
    return kron(
        AnalogRot(
            duration=duration,
            delta=-tower_detuning * tower_coeffs[i],
            phase=0.0,
            qubit_support=(i,),
        )
        for i in range(n_qubits)
    )


def analog_feature_map(
    param: str = "phi",
    op: Callable[[Parameter | Basic], AnalogBlock] = AnalogRX,
    fm_type: BasisSet | type[Function] | str = BasisSet.FOURIER,
    feature_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
    multiplier: Parameter | TParameter | None = None,
) -> AnalogBlock:
    """Generate a fully analog feature map.

    Args:
        param: Parameter of the feature map; you can pass a string or Parameter;
            it will be set as non-trainable (FeatureParameter) regardless.
        op: type of operation. Choose among AnalogRX, AnalogRY, AnalogRZ or a custom
            callable function returning an AnalogBlock instance
        fm_type: Basis set for data encoding; choose from `BasisSet.FOURIER` for Fourier
            encoding, or `BasisSet.CHEBYSHEV` for Chebyshev polynomials of the first kind.
        feature_range: range of data that the input data is assumed to come from.
        target_range: range of data the data encoder assumes as the natural range. For example,
            in Chebyshev polynomials it is (-1, 1), while for Fourier it may be chosen as (0, 2*pi).
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
            different scalings; can be a number or parameter/expression.
    """

    transformed_feature = fm_parameter(
        fm_type, param, feature_range=feature_range, target_range=target_range
    )
    multiplier = 1 if multiplier is None else multiplier
    return op(multiplier * transformed_feature)
