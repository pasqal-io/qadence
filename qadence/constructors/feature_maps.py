from __future__ import annotations

from typing import Type, Union

import numpy as np
import sympy

from qadence.blocks import AbstractBlock, KronBlock, chain, kron, tag
from qadence.operations import RX, RY, RZ, H
from qadence.parameters import FeatureParameter, Parameter

Rotation = Union[RX, RY, RZ]


def feature_map(
    n_qubits: int,
    support: tuple[int, ...] = None,
    param: str = "phi",
    op: Type[Rotation] = RX,
    fm_type: str = "fourier",
) -> KronBlock:
    """Construct a feature map of a given type.

    Arguments:
        n_qubits: Number of qubits the feature map covers. Results in `support=range(n_qubits)`.
        support: Overrides `n_qubits`. Puts one rotation gate on every qubit in `support`.
        param: Parameter of the feature map.
        op: Rotation operation of the feature map.
        fm_type: Determines the additional expression the final feature parameter (the addtional
            term in front of `param`). `"fourier": param` (nothing is done to `param`)
            `"chebyshev": 2*acos(param)`, `"tower": (i+1)*2*acos(param)` (where `i` is the qubit
            index).

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
    fparam = FeatureParameter(param)
    if support is None:
        support = tuple(range(n_qubits))

    assert len(support) <= n_qubits, "Wrong qubit support supplied"

    if fm_type == "fourier":
        fm = kron(*[op(qubit, fparam) for qubit in support])
    elif fm_type == "chebyshev":
        fm = kron(*[op(qubit, 2 * sympy.acos(fparam)) for qubit in support])
    elif fm_type == "tower":
        fm = kron(*[op(qubit, (i + 1) * 2 * sympy.acos(fparam)) for i, qubit in enumerate(support)])
    else:
        raise NotImplementedError(f"Feature map {fm_type} not implemented")
    fm.tag = "FM"
    return fm


def fourier_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: Type[Rotation] = RX
) -> AbstractBlock:
    """Construct a Fourier feature map

    Args:
        n_qubits: number of qubits across which the FM is created
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type="fourier")
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
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type="chebyshev")
    return tag(fm, tag="ChebyshevFM")


def tower_feature_map(
    n_qubits: int, support: tuple[int, ...] = None, param: str = "phi", op: Type[Rotation] = RX
) -> AbstractBlock:
    """Construct a Chebyshev tower feature map

    Args:
        n_qubits: number of qubits across which the FM is created
        param: The base name for the feature `Parameter`
    """
    fm = feature_map(n_qubits, support=support, param=param, op=op, fm_type="tower")
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

    if support is None:
        support = tuple(range(n_qubits))

    xmax = max(feature_range)
    xmin = min(feature_range)

    x = Parameter(param, trainable=False)

    # The feature map works on the range of 0 to 2**n
    x_rescaled = 2 * np.pi * (x - xmin) / (xmax - xmin)

    hlayer = kron(H(qubit) for qubit in support)
    rlayer = kron(RZ(support[i], x_rescaled * (2**i)) for i in range(n_qubits))

    return tag(chain(hlayer, rlayer), f"ExpFourierFM({param})")
