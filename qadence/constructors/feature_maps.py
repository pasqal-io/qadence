from __future__ import annotations

import inspect
import math
from typing import Callable, Type, Union

import sympy

from qadence.blocks import AbstractBlock, KronBlock, chain, kron, tag
from qadence.logger import get_logger
from qadence.operations import PHASE, RX, RY, RZ, H
from qadence.parameters import FeatureParameter, Parameter
from qadence.types import BasisSet, ReuploadScaling, TParameter

logger = get_logger(__name__)

TRotation = Type[Union[RX, RY, RZ, PHASE]]
ROTATIONS = [RX, RY, RZ, PHASE]


def _set_range(fm_type: BasisSet | Type[sympy.Function] | str) -> tuple[float, float]:
    if fm_type == BasisSet.FOURIER:
        return (0.0, 2 * sympy.pi)
    elif fm_type == BasisSet.CHEBYSHEV:
        return (-1.0, 1.0)
    else:
        return (0.0, 1.0)


def _rs_constant(i: int) -> int:
    return 1


def _rs_tower(i: int) -> float:
    return float(i + 1)


def _rs_exp(i: int) -> float:
    return float(2**i)


RS_FUNC_DICT = {
    ReuploadScaling.CONSTANT: _rs_constant,
    ReuploadScaling.TOWER: _rs_tower,
    ReuploadScaling.EXP: _rs_exp,
}


def feature_map(
    n_qubits: int,
    support: tuple[int, ...] | None = None,
    param: Parameter | sympy.Basic | str = "phi",
    op: TRotation = RX,
    fm_type: BasisSet | Type[sympy.Function] | str = BasisSet.FOURIER,
    reupload_scaling: ReuploadScaling | Callable | str = ReuploadScaling.CONSTANT,
    feature_range: tuple[float, float] | None = None,
    target_range: tuple[float, float] | None = None,
    multiplier: Parameter | TParameter | None = None,
) -> KronBlock:
    """Construct a feature map of a given type.

    Arguments:
        n_qubits: Number of qubits the feature map covers. Results in `support=range(n_qubits)`.
        support: Puts one feature-encoding rotation gate on every qubit in `support`. n_qubits in
            this case specifies the total overall qubits of the circuit, which may be wider than the
            support itself, but not narrower.
        param: Parameter of the feature map; you can pass a string, sympy expression or Parameter;
            it will be set as non-trainable (FeatureParameter) regardless.
        op: Rotation operation of the feature map; choose from RX, RY, RZ, PHASE
        fm_type: Determines the basis set for the encoding; choose from `BasisSet.FOURIER` for
            Fourier encoding, or `BasisSet.CHEBYSHEV` for Chebyshev polynomials of the first kind.
        reupload_scaling: how the feature map scales the data that is re-uploaded for each qubit.
        feature_range: range of data that the input data is assumed to come from.
        target_range: range of data the data encoder assumes as the natural range. For example,
            in Chebyshev polynomials it is (-1, 1), while for Fourier it may be chosen as (0, 2pi).
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
            different scalings; can be a number or parameter/expression.

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
            f"Operation {op} not supported. Please one from {[rot.__name__ for rot in ROTATIONS]}."
        )

    # Backward compatibility
    if fm_type in ("fourier", "chebyshev", "tower"):
        logger.warning(
            "Selecting `fm_type` as 'fourier', 'chebyshev' or 'tower' is deprecated. "
            "Please use the respective enumerations: 'fm_type = BasisSet.FOURIER', "
            "'fm_type = BasisSet.CHEBYSHEV' or 'reupload_scaling = ReuploadScaling.TOWER'."
        )
        fm_type = BasisSet.FOURIER if fm_type == "fourier" else fm_type
        fm_type = BasisSet.CHEBYSHEV if fm_type == "chebyshev" else fm_type
        reupload_scaling = ReuploadScaling.TOWER if fm_type == "tower" else reupload_scaling

    if isinstance(param, Parameter):
       fparam = param
       if fparam.trainable:
           fparam.trainable = False
    else:
       fparam = FeatureParameter(param)

    # Set feature and target range
    feature_range = _set_range(fm_type) if feature_range is None else feature_range
    target_range = _set_range(fm_type) if target_range is None else target_range

    # Rescale the feature parameter
    f_max = max(feature_range)
    f_min = min(feature_range)
    t_max = max(target_range)
    t_min = min(target_range)

    scaling = (t_max - t_min) / (f_max - f_min)
    shift = t_min - f_min * scaling

    if math.isclose(scaling, 1.0):
        # So we don't get 1.0 factor in visualization
        scaled_fparam = fparam + shift
    else:
        scaled_fparam = scaling * fparam + shift

    # Transform feature parameter
    if fm_type == BasisSet.FOURIER:
        transformed_feature = scaled_fparam
    elif fm_type == BasisSet.CHEBYSHEV:
        transformed_feature = sympy.acos(scaled_fparam)
    elif inspect.isclass(fm_type) and issubclass(fm_type, sympy.Function):
        transformed_feature = fm_type(scaled_fparam)
    else:
        raise NotImplementedError(
            f"Feature map type {fm_type} not implemented. Choose an item from the BasisSet enum: "
            f"{[bs.name for bs in BasisSet]}, or your own sympy.Function to wrap the given "
            "feature parameter with."
        )

    basis_tag = fm_type.value if isinstance(fm_type, BasisSet) else str(fm_type)

    # Set reupload scaling function
    if callable(reupload_scaling):
        rs_func = reupload_scaling
        rs_tag = "Custom"
    else:
        try:
            rs_func = RS_FUNC_DICT[reupload_scaling]  # type: ignore [index]
            if isinstance(reupload_scaling, ReuploadScaling):
                rs_tag = reupload_scaling.value
            else:
                rs_tag = reupload_scaling
        except (KeyError, ValueError) as error:
            raise NotImplementedError(
                f"Reupload scaling {reupload_scaling} not implemented; choose an item from "
                f"the ReuploadScaling enum: {[rs.name for rs in ReuploadScaling]}, or your own "
                "python function with a single int arg as input and int or float output."
            )

    # Set overall multiplier
    multiplier = 1 if multiplier is None else multiplier

    # Build feature map
    op_list = []
    for i, qubit in enumerate(support):
        op_list.append(op(qubit, multiplier * rs_func(i) * transformed_feature))
    fm = kron(*op_list)

    fm.tag = rs_tag + " " + basis_tag + " FM"

    return fm


def fourier_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: TRotation = RX
) -> AbstractBlock:
    """Construct a Fourier feature map.

    Args:
        n_qubits: number of qubits across which the FM is created
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type=BasisSet.FOURIER)
    return fm


def chebyshev_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: TRotation = RX
) -> AbstractBlock:
    """Construct a Chebyshev feature map.

    Args:
        n_qubits: number of qubits across which the FM is created
        support (Iterable[int]): The qubit support
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type=BasisSet.CHEBYSHEV)
    return fm


def tower_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: TRotation = RX
) -> AbstractBlock:
    """Construct a Chebyshev tower feature map.

    Args:
        n_qubits: number of qubits across which the FM is created
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(
        n_qubits,
        support=support,
        param=param,
        op=op,
        fm_type=BasisSet.CHEBYSHEV,
        reupload_scaling=ReuploadScaling.TOWER,
    )
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
        target_range=(0.0, 2 * sympy.pi),
    )
    rlayer.tag = None
    return tag(chain(hlayer, rlayer), f"ExpFourierFM({param})")
