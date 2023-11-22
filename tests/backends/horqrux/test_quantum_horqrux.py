from __future__ import annotations

import jax.numpy as jnp
import pytest
import torch

from qadence import (
    CNOT,
    CRX,
    CRY,
    CRZ,
    RX,
    RY,
    RZ,
    BackendName,
    X,
    Y,
    Z,
    expectation,
    run,
)
from qadence.backends.utils import jarr_to_tensor
from qadence.blocks import AbstractBlock
from qadence.constructors import hea


def test_psr() -> None:
    values = {"theta": torch.tensor([torch.pi])}
    psr = expectation(hea(2, 1), Z(0), values=values, backend="horqrux", diff_mode="gpsr")
    ad = expectation(hea(2, 1), Z(0), values=values, backend="horqrux", diff_mode="ad")
    breakpoint()
    assert jnp.allclose(psr[0], ad[0])


@pytest.mark.parametrize("block", [RX(0, 1.0), RY(0, 3.0), RZ(0, 4.0), X(0), Y(0), Z(0)])
def test_singlequbit_primitive_parametric(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize(
    "block", [CNOT(0, 1), CNOT(1, 0), CRX(0, 1, 1.0), CRY(1, 0, 2.0), CRZ(0, 1, 3.0)]
)
def test_control(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize("block", [hea(2, 1), hea(4, 4)])
def test_hea(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize("block", [hea(2, 1), hea(4, 4)])
def test_hea_expectation(block: AbstractBlock) -> None:
    exp_pyq = expectation(block, Z(0), backend=BackendName.PYQTORCH)
    exp_horq = jarr_to_tensor(
        expectation(block, Z(0), backend=BackendName.HORQRUX), dtype=torch.double
    )
    torch.allclose(exp_pyq, exp_horq)
