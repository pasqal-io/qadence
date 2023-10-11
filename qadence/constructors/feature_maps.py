from __future__ import annotations

import inspect
from enum import Enum
from typing import Callable, Type, Union

import sympy

from qadence.blocks import AbstractBlock, KronBlock, chain, kron, tag
from qadence.operations import PHASE, RX, RY, RZ, H
from qadence.parameters import FeatureParameter, Parameter
from qadence.types import TParameter

Rotation = Union[RX, RY, RZ, PHASE]


class BasisSet(str, Enum):
    FOURIER = "Fourier"
    CHEBYSHEV = "Chebyshev"


class ReuploadScaling(str, Enum):
    CONSTANT = "Constant"
    TOWER = "Tower"
    EXP_UP = "Exponential_up"
    EXP_DOWN = "Exponential_down"


def _set_range(fm_type: BasisSet | Type[sympy.Function] | str) -> tuple[float, float]:
    if fm_type == BasisSet.FOURIER:
        return (0.0, 2 * sympy.pi)
    elif fm_type == BasisSet.CHEBYSHEV:
        return (-1.0, 1.0)
    else:
        return (0.0, 1.0)


def _get_rs_func(
    reupload_scaling: Callable | ReuploadScaling, support: tuple[int, ...]
) -> tuple[Callable, str]:
    if reupload_scaling == ReuploadScaling.CONSTANT:

        def rs_func(i: int) -> float:
            return float(1)

    elif reupload_scaling == ReuploadScaling.TOWER:

        def rs_func(i: int) -> float:
            return float(i + 1)

    elif reupload_scaling == ReuploadScaling.EXP_UP:

        def rs_func(i: int) -> float:
            return float(2**i)

    elif reupload_scaling == ReuploadScaling.EXP_DOWN:

        def rs_func(i: int) -> float:
            return float(2 ** (len(support) - 1 - i))

    else:
        raise NotImplementedError(
            f"Re-upload scaling {reupload_scaling} not Implemented; choose one from /"
            f"{[rs for rs in ReuploadScaling]}, or your own python function with a /"
            "single int arg as input and int or float output!"
        )

    rs_tag = reupload_scaling.value

    return rs_func, rs_tag


def feature_map(
    n_qubits: int,
    support: tuple[int, ...] | None = None,
    param: Parameter | sympy.Basic | str = "phi",
    op: Type[Rotation] = RX,
    fm_type: BasisSet | Type[sympy.Function] | str = BasisSet.FOURIER,
    reupload_scaling: Callable | ReuploadScaling = ReuploadScaling.CONSTANT,
    feature_range: tuple[float, float] = None,
    target_range: tuple[float, float] = None,
    multiplier: Parameter | TParameter = None,
) -> KronBlock:
    """Construct a feature map of a given type.

    Arguments:
        n_qubits: Number of qubits the feature map covers. Results in `support=range(n_qubits)`.
        support: Puts one feature-encoding rotation gate on every qubit in `support`. n_qubits in
            this case specifies the total overall qubits of the circuit, which may be wider than the
            support itself, but not narrower.
        param: Parameter of the feature map; you can pass a string or sympy expression or Parameter;
            it will be set as non-trainable (FeatureParameter) regardless
        op: Rotation operation of the feature map; choose from RX, RY, RZ, PHASE
        fm_type: Determines the basis set that this encoding relates to; choose
                from `BasisSet.FOURIER` for Fourier encoding, or `BasisSet.CHEBYSHEV`
                for Chebyshev polynomials of the first kind.
        reupload_scaling: the feature_map re-uploads the data on len(support) qubits;
            however, in each re-upload one may choose a different scaling factor in front,
            to enrich the spectrum.
        feature_range: what is the range of data that the input data is assumed to come from?
        target_range: what is the range of data the data encoder assumes as the natural range? for
            example, in Chebyshev polynomials, the natural range is (-1,1), while for Fourier it may
             be chosen as (0,2pi)
        multiplier: overall multiplier; this is useful for reuploading the feature map serially with
         different scalings there; can be a number or parameter/expression

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence import feature_map

    fm = feature_map(3, fm_type="fourier")
    print(f"{fm = }")

    fm = feature_map(3, fm_type="chebyshev")
    print(f"{fm = }")

    fm = feature_map(3, fm_type="tower")
    print(f"{fm = }")
    ```
    """
    if isinstance(param, Parameter):
        fparam = param
        if fparam.is_trainable:
            fparam.is_trainable = False
    else:
        fparam = FeatureParameter(param)

    if support is None:
        support = tuple(range(n_qubits))

    assert len(support) <= n_qubits, "Wrong qubit support supplied"

    # Set feature and target range
    feature_range = _set_range(fm_type) if feature_range is None else feature_range
    target_range = _set_range(fm_type) if target_range is None else target_range

    # Rescale the parameter according to feature and target range
    f_max = max(feature_range)
    f_min = min(feature_range)
    t_max = max(target_range)
    t_min = min(target_range)

    scaled_fparam = (t_max - t_min) * (fparam - f_min) / (f_max - f_min) + t_max

    if fm_type == BasisSet.FOURIER:
        transformed_feature = scaled_fparam
        basis_tag = fm_type.value
    elif fm_type == BasisSet.CHEBYSHEV:
        transformed_feature = sympy.acos(scaled_fparam)
        basis_tag = fm_type.value
    elif inspect.isclass(fm_type) and issubclass(fm_type, sympy.Function):
        transformed_feature = fm_type(scaled_fparam)
        basis_tag = str(fm_type)
    else:
        raise NotImplementedError(
            f"{fm_type} not implemented. Choose a basis set from {[bs for bs in BasisSet]}, /"
            "or your own sympy.Function to wrap the given feature parameter with."
        )

    # Set reupload scaling function
    if callable(reupload_scaling):
        rs_func = reupload_scaling
        rs_tag = "Custom"
    else:
        rs_func, rs_tag = _get_rs_func(reupload_scaling, support)

    # Set multiplier
    mult = 1.0 if multiplier is None else multiplier

    # Build feature map
    op_list = []
    for i, qubit in enumerate(support):
        op_list.append(op(qubit, mult * rs_func(i) * transformed_feature))  # type: ignore[operator]
    fm = kron(*op_list)

    fm.tag = rs_tag + " " + basis_tag + " FM"
    return fm


def fourier_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: Type[Rotation] = RX
) -> AbstractBlock:
    """Construct a Fourier feature map

    Args:
        n_qubits: number of qubits across which the FM is created
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type=BasisSet.FOURIER)
    return tag(fm, tag="FourierFM")


def chebyshev_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: Type[Rotation] = RX
) -> AbstractBlock:
    """Construct a Chebyshev feature map

    Args:
        n_qubits: number of qubits across which the FM is created
        support (Iterable[int]): The qubit support
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type=BasisSet.CHEBYSHEV)
    return tag(fm, tag="ChebyshevFM")


def tower_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: Type[Rotation] = RX
) -> AbstractBlock:
    """Construct a Chebyshev tower feature map

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
    return tag(fm, tag="TowerFM")


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
        reupload_scaling=ReuploadScaling.EXP_UP,
        feature_range=feature_range,
        target_range=(0.0, 2 * sympy.pi),
    )
    rlayer.tag = None
    return tag(chain(hlayer, rlayer), f"ExpFourierFM({param})")
