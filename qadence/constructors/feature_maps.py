from __future__ import annotations

from collections.abc import Callable
from logging import getLogger
from math import isclose
from typing import Union

from sympy import Basic, acos

from qadence.blocks import AbstractBlock, KronBlock, chain, kron, tag
from qadence.operations import PHASE, RX, RY, RZ, H
from qadence.parameters import FeatureParameter, Parameter, VariationalParameter
from qadence.types import PI, BasisSet, ReuploadScaling, TParameter

logger = getLogger(__name__)

ROTATIONS = [RX, RY, RZ, PHASE]
RotationTypes = type[Union[RX, RY, RZ, PHASE]]


def _set_range(fm_type: BasisSet | Callable | str) -> tuple[float, float]:
    if fm_type == BasisSet.FOURIER:
        return (0.0, 2 * PI)
    elif fm_type == BasisSet.CHEBYSHEV:
        return (-1.0, 1.0)
    else:
        return (0.0, 1.0)


RS_FUNC_DICT = {
    ReuploadScaling.CONSTANT: lambda i: 1,
    ReuploadScaling.TOWER: lambda i: float(i + 1),
    ReuploadScaling.EXP: lambda i: float(2**i),
}


def fm_parameter_scaling(
    fm_type: BasisSet | Callable | str,
    param: Parameter | str = "phi",
    feature_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
) -> Parameter | Basic:
    if isinstance(param, Parameter):
        fparam = param
        fparam.trainable = False
    else:
        fparam = FeatureParameter(param)

    # Set feature and target range
    feature_range = _set_range(fm_type) if feature_range is None else feature_range
    target_range = _set_range(fm_type) if target_range is None else target_range

    # Rescale the feature parameter
    scaling = (max(target_range) - min(target_range)) / (max(feature_range) - min(feature_range))
    shift = min(target_range) - min(feature_range) * scaling

    if isclose(scaling, 1.0):
        # So we don't get 1.0 factor in visualization
        scaled_fparam = fparam + shift
    else:
        scaled_fparam = scaling * fparam + shift

    return scaled_fparam


def fm_parameter_func(fm_type: BasisSet | Callable | str) -> Callable:
    def ident_fn(x: TParameter) -> TParameter:
        return x

    # Transform feature parameter
    if fm_type == BasisSet.FOURIER:
        transform_func = ident_fn
    elif fm_type == BasisSet.CHEBYSHEV:
        transform_func = acos
    elif callable(fm_type):
        transform_func = fm_type
    else:
        raise NotImplementedError(
            f"Feature map type {fm_type} not implemented. Choose an item from the BasisSet "
            f"enum: {BasisSet.list()}, or a custom defined sympy function to wrap "
            "the given feature parameter with."
        )

    return transform_func


def fm_reupload_scaling_fn(
    reupload_scaling: ReuploadScaling | Callable | str = ReuploadScaling.CONSTANT,
) -> tuple[Callable, str]:
    # Set reupload scaling function
    if callable(reupload_scaling):
        rs_func = reupload_scaling
        rs_tag = "Custom"
    else:
        rs_func = RS_FUNC_DICT.get(reupload_scaling, None)  # type: ignore [call-overload]
        if rs_func is None:
            raise NotImplementedError(
                f"Reupload scaling {reupload_scaling} not implemented; choose an item from "
                f"the ReuploadScaling enum: {[rs.name for rs in ReuploadScaling]}, or your own "
                "python function with a single int arg as input and int or float output."
            )
        if isinstance(reupload_scaling, ReuploadScaling):
            rs_tag = reupload_scaling.value
        else:
            rs_tag = reupload_scaling

    return rs_func, rs_tag


def feature_map(
    n_qubits: int,
    support: tuple[int, ...] | None = None,
    param: Parameter | str = "phi",
    op: RotationTypes = RX,
    fm_type: BasisSet | Callable | str = BasisSet.FOURIER,
    reupload_scaling: ReuploadScaling | Callable | str = ReuploadScaling.CONSTANT,
    feature_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
    multiplier: Parameter | TParameter | None = None,
    param_prefix: str | None = None,
) -> KronBlock:
    """Construct a feature map of a given type.

    Arguments:
        n_qubits: Number of qubits the feature map covers. Results in `support=range(n_qubits)`.
        support: Puts one feature-encoding rotation gate on every qubit in `support`. n_qubits in
            this case specifies the total overall qubits of the circuit, which may be wider than the
            support itself, but not narrower.
        param: Parameter of the feature map; you can pass a string or Parameter;
            it will be set as non-trainable (FeatureParameter) regardless.
        op: Rotation operation of the feature map; choose from RX, RY, RZ or PHASE.
        fm_type: Basis set for data encoding; choose from `BasisSet.FOURIER` for Fourier
            encoding, or `BasisSet.CHEBYSHEV` for Chebyshev polynomials of the first kind.
        reupload_scaling: how the feature map scales the data that is re-uploaded for each qubit.
            choose from `ReuploadScaling` enumeration or provide your own function with a single
            int as input and int or float as output.
        feature_range: range of data that the input data provided comes from. Used to map input data
            to the correct domain of the feature-encoding function.
        target_range: range of data the data encoder assumes as the natural range. For example,
            in Chebyshev polynomials it is (-1, 1), while for Fourier it may be chosen as (0, 2*PI).
            Used to map data to the correct domain of the feature-encoding function.
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
            different scalings; can be a number or parameter/expression.
        param_prefix: string prefix to create trainable parameters multiplying the feature parameter
            inside the feature-encoding function. Note that currently this does not take into
            account the domain of the feature-encoding function.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import feature_map, BasisSet, ReuploadScaling

    fm = feature_map(3, fm_type=BasisSet.FOURIER)
    print(f"{fm = }")

    fm = feature_map(3, fm_type=BasisSet.CHEBYSHEV)
    print(f"{fm = }")

    fm = feature_map(3, fm_type=BasisSet.FOURIER, reupload_scaling = ReuploadScaling.TOWER)
    print(f"{fm = }")
    ```
    """

    # Process input
    if support is None:
        support = tuple(range(n_qubits))
    elif len(support) != n_qubits:
        raise ValueError("Wrong qubit support supplied")

    if op not in ROTATIONS:
        raise ValueError(
            f"Operation {op} not supported. "
            f"Please provide one from {[rot.__name__ for rot in ROTATIONS]}."
        )

    scaled_fparam = fm_parameter_scaling(
        fm_type, param, feature_range=feature_range, target_range=target_range
    )

    transform_func = fm_parameter_func(fm_type)

    basis_tag = fm_type.value if isinstance(fm_type, BasisSet) else str(fm_type)
    rs_func, rs_tag = fm_reupload_scaling_fn(reupload_scaling)

    # Set overall multiplier
    multiplier = 1 if multiplier is None else Parameter(multiplier)

    # Build feature map
    op_list = []
    fparam = scaled_fparam
    for i, qubit in enumerate(support):
        if param_prefix is not None:
            train_param = VariationalParameter(param_prefix + f"_{i}")
            fparam = train_param * scaled_fparam
        op_list.append(op(qubit, multiplier * rs_func(i) * transform_func(fparam)))
    fm = kron(*op_list)

    fm.tag = rs_tag + " " + basis_tag + " FM"

    return fm


def exp_fourier_feature_map(
    n_qubits: int,
    support: tuple[int, ...] = None,
    param: str = "x",
    feature_range: tuple[float, float] = None,
) -> AbstractBlock:
    """
    Exponential fourier feature map.

    Args:
        n_qubits: number of qubits in the feature
        support: qubit support
        param: name of feature `Parameter`
        feature_range: min and max value of the feature, as floats in a Tuple
    """

    if feature_range is None:
        feature_range = (0.0, 2.0**n_qubits)

    support = tuple(range(n_qubits)) if support is None else support
    hlayer = kron(H(qubit) for qubit in support)
    rlayer = feature_map(
        n_qubits,
        support=support,
        param=param,
        op=RZ,
        fm_type=BasisSet.FOURIER,
        reupload_scaling=ReuploadScaling.EXP,
        feature_range=feature_range,
        target_range=(0.0, 2 * PI),
    )
    rlayer.tag = None
    return tag(chain(hlayer, rlayer), f"ExpFourierFM({param})")
