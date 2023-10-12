from __future__ import annotations

from typing import Callable

import pytest
import sympy
import torch
from metrics import ATOL_64

from qadence import (
    PHASE,
    RX,
    BasisSet,
    FeatureParameter,
    ReuploadScaling,
    Z,
    expectation,
    feature_map,
    run,
)

PARAM_DICT_0 = {
    "support": None,
    "param": FeatureParameter("x"),
    "op": RX,
    "feature_range": None,
    "multiplier": None,
}

PARAM_DICT_1 = {
    "support": (3, 2, 1, 0),
    "param": "x",
    "op": PHASE,
    "feature_range": (-2.0, -1.0),
    "target_range": (1.0, 5.0),
    "multiplier": FeatureParameter("y"),
}


@pytest.mark.parametrize("param_dict", [PARAM_DICT_0, PARAM_DICT_1])
@pytest.mark.parametrize("fm_type", [BasisSet.FOURIER, BasisSet.CHEBYSHEV, sympy.asin])
@pytest.mark.parametrize(
    "reupload_scaling",
    [ReuploadScaling.CONSTANT, ReuploadScaling.TOWER, ReuploadScaling.EXP, lambda i: 5 * i + 2],
)
def test_feature_map_creation_and_run(
    param_dict: dict,
    fm_type: BasisSet | type[sympy.Function],
    reupload_scaling: ReuploadScaling | Callable,
) -> None:
    n_qubits = 4

    block = feature_map(
        n_qubits=n_qubits, fm_type=fm_type, reupload_scaling=reupload_scaling, **param_dict
    )

    values = {"x": torch.rand(1), "y": torch.rand(1)}

    run(block, values=values)


@pytest.mark.parametrize("fm_type", [BasisSet.FOURIER, BasisSet.CHEBYSHEV])
@pytest.mark.parametrize(
    "reupload_scaling",
    [ReuploadScaling.TOWER, ReuploadScaling.CONSTANT, ReuploadScaling.EXP, "exp_down"],
)
def test_feature_map_correctness(fm_type: BasisSet, reupload_scaling: ReuploadScaling) -> None:
    n_qubits = 4
    support = tuple(range(n_qubits))

    # Preparing exact result
    if fm_type == BasisSet.CHEBYSHEV:
        transform = torch.acos
        xv = torch.linspace(-0.95, 0.95, 100)
        feature_range = (-1.0, 1.0)
        target_range = (-1.0, 1.0)
    elif fm_type == BasisSet.FOURIER:

        def transform(x: torch.Tensor) -> torch.Tensor:
            return x

        xv = torch.linspace(0.0, 2 * torch.pi, 100)
        feature_range = (0.0, 2 * torch.pi)
        target_range = (0.0, 2 * torch.pi)

    if reupload_scaling == ReuploadScaling.CONSTANT:

        def scaling(j: int) -> float:
            return 1

    elif reupload_scaling == ReuploadScaling.TOWER:

        def scaling(j: int) -> float:
            return float(j + 1)

    elif reupload_scaling == ReuploadScaling.EXP:

        def scaling(j: int) -> float:
            return float(2**j)

    elif reupload_scaling == "exp_down":

        def scaling(j: int) -> float:
            return float(2 ** (n_qubits - j - 1))

        reupload_scaling = ReuploadScaling.EXP
        support = tuple(reversed(range(n_qubits)))

    target = torch.cat(
        [torch.cos(scaling(j) * transform(xv)).unsqueeze(1) for j in range(n_qubits)], 1
    )

    # Running the block expectation
    block = feature_map(
        n_qubits=n_qubits,
        support=support,
        param="x",
        op=RX,
        fm_type=fm_type,
        reupload_scaling=reupload_scaling,
        feature_range=feature_range,
        target_range=target_range,
    )

    yv = expectation(block, [Z(j) for j in range(n_qubits)], values={"x": xv})

    # Assert correctness
    assert torch.allclose(yv, target, atol=ATOL_64)
