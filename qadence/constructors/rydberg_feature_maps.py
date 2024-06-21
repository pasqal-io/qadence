from __future__ import annotations

from logging import getLogger
from typing import Callable

import numpy as np
from sympy import Basic

from qadence.blocks import AnalogBlock, KronBlock, kron
from qadence.constructors.feature_maps import fm_parameter_func, fm_parameter_scaling
from qadence.operations import AnalogRot, AnalogRX, AnalogRY, AnalogRZ
from qadence.parameters import FeatureParameter, Parameter, VariationalParameter
from qadence.types import PI, BasisSet, ReuploadScaling, TParameter

logger = getLogger(__name__)

AnalogRotationTypes = [AnalogRX, AnalogRY, AnalogRZ]


def rydberg_feature_map(
    n_qubits: int,
    param: str = "phi",
    max_abs_detuning: float = 2 * PI * 10,
    weights: list[float] | None = None,
) -> KronBlock:
    """Feature map using semi-local addressing patterns.

    If not weights are specified, variational parameters are created
    for the pattern

    Args:
        n_qubits (int): number of qubits
        param: the name of the feature parameter
        max_abs_detuning: maximum value of absolute detuning for each qubit. Defaulted at 10 MHz.
        weights: a list of wegiths to assign to each qubit parameter in the feature map

    Returns:
        The block representing the feature map
    """

    tower_coeffs: list[float | Parameter]
    tower_coeffs = (
        [VariationalParameter(f"w_{param}_{i}") for i in range(n_qubits)]
        if weights is None
        else weights
    )
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


def rydberg_tower_feature_map(
    n_qubits: int, param: str = "phi", max_abs_detuning: float = 2 * PI * 10
) -> KronBlock:
    weights = list(np.arange(1, n_qubits + 1))
    return rydberg_feature_map(
        n_qubits, param=param, max_abs_detuning=max_abs_detuning, weights=weights
    )


def analog_feature_map(
    param: str = "phi",
    op: Callable[[Parameter | Basic], AnalogBlock] = AnalogRX,
    fm_type: BasisSet | Callable | str = BasisSet.FOURIER,
    reupload_scaling: ReuploadScaling | Callable | str = ReuploadScaling.CONSTANT,
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
        reupload_scaling: how the feature map scales the data that is re-uploaded. Given that
            this feature map uses analog rotations, the reuploading works by simply
            adding additional operations with different scaling factors in the parameter.
            Choose from `ReuploadScaling` enumeration, currently only CONSTANT works,
            or provide your own function with the first argument being the given
            operation `op` and the second argument the feature parameter
        feature_range: range of data that the input data is assumed to come from.
        target_range: range of data the data encoder assumes as the natural range. For example,
            in Chebyshev polynomials it is (-1, 1), while for Fourier it may be chosen as (0, 2*PI).
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
            different scalings; can be a number or parameter/expression.
    """

    scaled_fparam = fm_parameter_scaling(
        fm_type, param, feature_range=feature_range, target_range=target_range
    )

    transform_func = fm_parameter_func(fm_type)

    transformed_feature = transform_func(scaled_fparam)

    multiplier = 1.0 if multiplier is None else Parameter(multiplier)

    if callable(reupload_scaling):
        return reupload_scaling(op, multiplier * transformed_feature)  # type: ignore[no-any-return]
    elif reupload_scaling == ReuploadScaling.CONSTANT:
        return op(multiplier * transformed_feature)
    # TODO: implement tower scaling by reuploading multiple times
    # using different analog rotations
    else:
        raise NotImplementedError(f"Reupload scaling {str(reupload_scaling)} not implemented!")
