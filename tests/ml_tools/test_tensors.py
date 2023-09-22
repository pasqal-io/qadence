from __future__ import annotations

import numpy as np
import pytest
import torch

from qadence.ml_tools import numpy_to_tensor, promote_to, promote_to_tensor


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_numpy_to_tensor(requires_grad: bool, dtype: torch.dtype) -> None:
    array_np = np.random.random((10, 2))
    array_tc = numpy_to_tensor(array_np, dtype=dtype, requires_grad=requires_grad)

    assert array_tc.requires_grad == requires_grad
    assert array_tc.dtype == dtype
    assert np.allclose(array_np, array_tc.detach().numpy())


@pytest.mark.parametrize("requires_grad", [True, False])
def test_promote_to_tensor(requires_grad: bool) -> None:
    array_np = np.linspace(0, 1, 100)
    array_tc = promote_to_tensor(array_np, requires_grad=requires_grad)

    assert array_tc.requires_grad == requires_grad
    assert np.allclose(array_np, array_tc.detach().numpy())

    number = 1.2345
    number_tc = promote_to_tensor(number, requires_grad=requires_grad)
    assert number_tc.requires_grad == requires_grad
    assert number_tc.shape == (1, 1)
    assert np.isclose(float(number_tc.flatten()), number)


def test_promote_to() -> None:
    array_tc = torch.linspace(0, 1, 100)
    array_np = promote_to(array_tc, np.ndarray)
    assert np.allclose(array_np, array_tc.detach().numpy())

    number_tc = torch.Tensor([1.2345]).reshape(-1, 1)
    number = promote_to(number_tc, float)
    assert np.isclose(float(number_tc.flatten()), number)

    array_tc = torch.rand(10, 2)
    array_tc_prom = promote_to(array_tc, torch.Tensor)
    assert torch.equal(array_tc, array_tc_prom)
