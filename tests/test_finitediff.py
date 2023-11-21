from itertools import combinations_with_replacement
import pytest
import torch
import qadence as qd
from qadence.finitediff import finitediff
from typing import Callable


def _torch_derivative(ufa: Callable, x: torch.Tensor, derivative_indices: tuple[int, ...]) -> torch.Tensor:
    y = ufa(x)
    for idx in derivative_indices:
        out = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        y = out[:, idx]
    return y.reshape(-1, 1)


@pytest.mark.parametrize("idxs", [ (0,), (1,), (0,0), (1,0), (0,1), (1,1), (0,0,0), (1,1,1), (0,1,0),
                             (1,1,0), (1,0,1,0)])
def test_finitediff(idxs: tuple) -> None:

    fm = qd.kron(
        qd.feature_map(2, support=(0,1), param="x"),
        qd.feature_map(2, support=(2,3), param="y")
    )

    ufa = qd.QNN(
        qd.QuantumCircuit(4, fm, qd.hea(4,2)),
        observable=qd.total_magnetization(4),
        diff_mode = "ad",
        inputs = ["x", "y"]
    )

    xs = torch.rand(5, 2, requires_grad=True)
    ys = ufa(xs)
    print(f"{finitediff(ufa, xs, idxs) = }")
    print(f"{_torch_derivative(ufa, xs, idxs) = }")
    assert torch.allclose(finitediff(ufa, xs, idxs), _torch_derivative(ufa, xs, idxs), atol=1e-3)
