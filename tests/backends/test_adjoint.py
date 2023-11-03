from __future__ import annotations

import pytest
import torch
from metrics import ADJOINT_ACCEPTANCE

from qadence.backends.api import backend_factory
from qadence.blocks import AbstractBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.constructors import hea
from qadence.operations import CPHASE, RX, HamEvo, X, Z
from qadence.parameters import VariationalParameter
from qadence.types import DiffMode

torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("diff_mode", [DiffMode.ADJOINT])
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

    assert torch.autograd.gradcheck(lambda x: func(x, inputs_y), inputs_x)
    assert torch.autograd.gradcheck(lambda y: func(inputs_x, y), inputs_y)


@pytest.mark.parametrize("diff_mode", [DiffMode.ADJOINT])
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

    assert torch.autograd.gradcheck(func, theta)


@pytest.mark.flaky
def test_hea_derivatives() -> None:
    n_qubits = 2
    observable: list[AbstractBlock] = [Z(0)]
    block = hea(n_qubits, 1)
    circ = QuantumCircuit(n_qubits, block)
    theta_0_value = torch.rand(1, requires_grad=True)

    def get_grad(theta: torch.Tensor, circ: QuantumCircuit, diff_mode: str) -> torch.Tensor:
        bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
        pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)
        param_name = "theta_0"
        params[param_name] = theta

        def func(theta: torch.Tensor) -> torch.Tensor:
            inputs = {param_name: theta}
            all_params = embeddings_fn(params, inputs)
            return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

        exp = bknd.expectation(
            pyqtorch_circ, pyqtorch_obs, embeddings_fn(params, {param_name: theta})
        )
        dydtheta = torch.autograd.grad(exp, theta, torch.ones_like(exp))[0]
        return dydtheta

    ad_grad = get_grad(theta_0_value, circ, "ad")
    adjoint_grad = get_grad(theta_0_value, circ, "adjoint")
    assert torch.allclose(ad_grad, adjoint_grad, atol=ADJOINT_ACCEPTANCE)


def test_hamevo_timeevo_grad() -> None:
    generator = X(0)
    fmx = HamEvo(generator, parameter=VariationalParameter("theta"))

    circ = QuantumCircuit(2, fmx)
    obs = Z(0)
    backend = backend_factory(backend="pyqtorch", diff_mode=DiffMode.ADJOINT)
    (pyqtorch_circ, pyqtorch_obs, embeddings_fn, params) = backend.convert(circ, obs)
    theta = torch.rand(1, requires_grad=True)

    def func(theta: torch.Tensor) -> torch.Tensor:
        inputs = {"theta": theta}
        all_params = embeddings_fn(params, inputs)
        return backend.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    assert torch.autograd.gradcheck(func, theta, nondet_tol=ADJOINT_ACCEPTANCE)


def test_hamevo_generator_grad() -> None:
    theta = VariationalParameter("theta")
    generator = RX(0, theta)
    fmx = HamEvo(generator, parameter=1.0)

    circ = QuantumCircuit(2, fmx)
    obs = Z(0)
    backend = backend_factory(backend="pyqtorch", diff_mode=DiffMode.ADJOINT)
    (pyqtorch_circ, pyqtorch_obs, embeddings_fn, params) = backend.convert(circ, obs)
    theta = torch.rand(1, requires_grad=True)

    def func(theta: torch.Tensor) -> torch.Tensor:
        inputs = {"theta": theta}
        all_params = embeddings_fn(params, inputs)
        return backend.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    assert torch.autograd.gradcheck(func, theta, nondet_tol=ADJOINT_ACCEPTANCE)
