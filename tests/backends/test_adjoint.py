from __future__ import annotations

from typing import Callable

import pytest
import torch

from qadence.backends.api import backend_factory
from qadence.blocks import AbstractBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.operations import CPHASE, RX, X, Z
from qadence.parameters import VariationalParameter
from qadence.types import DiffMode


@pytest.mark.parametrize("diff_mode", list(DiffMode))
def test_pyq_differentiation(diff_mode: str) -> None:
    batch_size = 1
    n_qubits = 2
    observable: list[AbstractBlock] = [Z(0)]
    circ = QuantumCircuit(n_qubits, chain(RX(0, "x"), CPHASE(0, 1, "y")))

    bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
    pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)

    inputs_x = torch.rand(batch_size, requires_grad=True)
    inputs_y = torch.rand(batch_size, requires_grad=True)

    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = {"x": x, "y": y}
        all_params = embeddings_fn(params, inputs)
        return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    expval = func(inputs_x, inputs_y)

    def finitediff(f: Callable, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        return (f(x + eps) - f(x - eps)) / (2 * eps)  # type: ignore

    assert torch.allclose(
        finitediff(lambda x: func(x, inputs_y), inputs_x),
        torch.autograd.grad(expval, inputs_x, torch.ones_like(expval), retain_graph=True)[0],
    )

    assert torch.allclose(
        finitediff(lambda y: func(inputs_x, y), inputs_y),
        torch.autograd.grad(expval, inputs_y, torch.ones_like(expval), retain_graph=True)[0],
    )


@pytest.mark.parametrize("diff_mode", [DiffMode.AD, DiffMode.ADJOINT])
def test_scale_derivatives(diff_mode: str) -> None:
    batch_size = 1
    n_qubits = 1
    observable: list[AbstractBlock] = [Z(0)]
    circ = QuantumCircuit(n_qubits, VariationalParameter("theta") * X(0))

    bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
    pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)

    theta = torch.rand(batch_size, requires_grad=True)

    def func(theta: torch.Tensor) -> torch.Tensor:
        inputs = {"theta": theta}
        all_params = embeddings_fn(params, inputs)
        return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    expval = func(theta)

    def finitediff(f: Callable, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        return (f(x + eps) - f(x - eps)) / (2 * eps)  # type: ignore

    dfdtheta = torch.autograd.grad(expval, theta, torch.ones_like(expval))[0]
    fin_diff = finitediff(lambda theta: func(theta), theta)
    assert torch.allclose(dfdtheta, fin_diff)
