from __future__ import annotations

import pytest
import torch
from metrics import ADJOINT_ACCEPTANCE

from qadence.backends.api import backend_factory
from qadence.backends.pyqtorch.convert_ops import dydx, dydxx
from qadence.blocks import AbstractBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea
from qadence.operations import CPHASE, RX, HamEvo, X, Z
from qadence.parameters import VariationalParameter
from qadence.types import DiffMode


@pytest.mark.parametrize("diff_mode", [DiffMode.ADJOINT])
def test_pyq_differentiation(diff_mode: str) -> None:
    batch_size = 1
    n_qubits = 2
    observable: list[AbstractBlock] = [Z(0)]
    circ = QuantumCircuit(n_qubits, chain(RX(0, 3 * "x"), CPHASE(0, 1, "y")))

    bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
    pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)

    inputs_x = torch.rand(batch_size, requires_grad=True)
    inputs_y = torch.rand(batch_size, requires_grad=True)

    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = {"x": x, "y": y}
        all_params = embeddings_fn(params, inputs)
        return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    assert torch.autograd.gradcheck(
        lambda x: func(x, inputs_y), inputs_x, nondet_tol=ADJOINT_ACCEPTANCE
    )
    assert torch.autograd.gradcheck(
        lambda y: func(inputs_x, y), inputs_y, nondet_tol=ADJOINT_ACCEPTANCE
    )


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
def test_higher_order_hea_derivatives() -> None:
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
        dydtheta = torch.autograd.grad(exp, theta, torch.ones_like(exp), create_graph=True)[0]
        dydthetatheta = torch.autograd.grad(
            dydtheta, theta, torch.ones_like(dydtheta), retain_graph=True, create_graph=True
        )[0]
        dydthetathetatheta = torch.autograd.grad(
            dydthetatheta, theta, dydthetatheta, retain_graph=True
        )[0]
        return dydtheta, dydthetatheta, dydthetathetatheta

    ad_grad, ad_gradgrad, ad_gradgradgrad = get_grad(theta_0_value, circ, "ad")
    adjoint_grad, adjoint_gradgrad, adjoint_gradgradgrad = get_grad(theta_0_value, circ, "adjoint")
    assert torch.allclose(ad_grad, adjoint_grad, atol=ADJOINT_ACCEPTANCE)
    assert torch.allclose(ad_gradgrad, adjoint_gradgrad, atol=ADJOINT_ACCEPTANCE)
    assert torch.allclose(ad_gradgradgrad, adjoint_gradgradgrad, atol=ADJOINT_ACCEPTANCE)


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


@pytest.mark.skip("To be implemented properly.")
def test_higher_order() -> None:
    batch_size = 1
    n_qubits = 1
    observable: list[AbstractBlock] = [Z(0)]
    circ = QuantumCircuit(n_qubits, chain(RX(0, "x")))

    bknd = backend_factory(backend="pyqtorch", diff_mode="ad")
    pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)

    inputs_x = torch.rand(batch_size, requires_grad=True)

    inputs = {"x": inputs_x}
    all_params = embeddings_fn(params, inputs)
    out_state = pyqtorch_circ.native.run(values=all_params)
    projected_state = pyqtorch_obs[0].native.run(out_state, all_params)
    op = pyqtorch_circ.native.operations[0].operations[0]
    with torch.no_grad():
        dydx_res = dydx(op.jacobian({"x": inputs_x}), op.qubit_support, out_state, projected_state)
        dydxx_res = dydxx(op, {"x": inputs_x}, out_state, projected_state)

    def func(x: torch.Tensor) -> torch.Tensor:
        inputs = {"x": x}
        all_params = embeddings_fn(params, inputs)
        return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    exp = func(inputs_x)
    grad = torch.autograd.grad(exp, inputs_x, torch.ones_like(exp), create_graph=True)[0]
    gradgrad = torch.autograd.grad(grad, inputs_x, torch.ones_like(grad), retain_graph=True)[0]
    assert torch.allclose(dydx_res, grad, atol=ADJOINT_ACCEPTANCE)
    assert torch.allclose(dydxx_res, gradgrad, atol=ADJOINT_ACCEPTANCE)


@pytest.mark.skip("Native adjoint higher order derivatives will added soon.")
def test_higher_order_fm_derivatives() -> None:
    n_qubits = 2
    observable: list[AbstractBlock] = [Z(0)]
    fm = feature_map(n_qubits)
    block = hea(n_qubits, 1)
    circ = QuantumCircuit(n_qubits, chain(fm, block))
    theta_0_value = torch.rand(1, requires_grad=True)
    phi_0_value = torch.rand(1, requires_grad=True)

    def get_grad(
        theta: torch.Tensor, phi: torch.Tensor, circ: QuantumCircuit, diff_mode: str
    ) -> torch.Tensor:
        bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
        pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)
        param_name = "theta_0"
        params[param_name] = theta

        fm_param_name = "phi"
        params[fm_param_name] = phi

        exp = bknd.expectation(
            pyqtorch_circ,
            pyqtorch_obs,
            embeddings_fn(params, {param_name: theta, fm_param_name: phi}),
        )

        dydphi, dydtheta = torch.autograd.grad(
            exp, (phi, theta), torch.ones_like(exp), create_graph=True
        )

        dydphidtheta = torch.autograd.grad(
            dydphi, theta, torch.ones_like(dydtheta), create_graph=True
        )[0]
        # dydphidtheta = torch.autograd.grad(dydphidtheta,theta,dydphidtheta,create_graph=True)[0]
        # dydphidphi = torch.autograd.grad(
        #     dydphi, phi, torch.ones_like(dydphi), retain_graph=True
        # )[0]
        # dydthetathetatheta = torch.autograd.grad(
        #     dydthetatheta, theta, dydthetatheta, retain_graph=True
        # )[0]
        return dydphi, dydphidtheta, None

    ad_grad, ad_gradgrad, ad_gradgradgrad = get_grad(theta_0_value, phi_0_value, circ, "ad")

    adjoint_grad, adjoint_gradgrad, adjoint_gradgradgrad = get_grad(
        theta_0_value, phi_0_value, circ, "adjoint"
    )
    assert torch.allclose(ad_grad, adjoint_grad, atol=ADJOINT_ACCEPTANCE)
    assert torch.allclose(ad_gradgrad, adjoint_gradgrad, atol=ADJOINT_ACCEPTANCE)
    # assert torch.allclose(ad_gradgradgrad, adjoint_gradgradgrad, atol=ADJOINT_ACCEPTANCE)
