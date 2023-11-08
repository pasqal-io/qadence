from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch

from qadence.backends.api import backend_factory
from qadence.backends.utils import finitediff
from qadence.blocks import AbstractBlock, add, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.operations import CNOT, RX, RZ, Z
from qadence.parameters import Parameter, VariationalParameter

torch.manual_seed(42)

expected_pi = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
expected_pi2 = torch.tensor([[0.5, 0.0], [0.5, 0.0]])


def parametric_circuit(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit"""

    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=False)

    fm = kron(RX(0, 3 * x), RZ(1, sympy.exp(y)), RX(2, 0.5), RZ(3, x))
    ansatz = kron(CNOT(0, 1), CNOT(2, 3))
    rotlayer1 = kron(RX(i, f"w_{i}") for i in range(n_qubits))

    theta = VariationalParameter("theta")
    rotlayer2 = kron(RX(i, 3.0 * theta) for i in range(n_qubits))

    block = chain(fm, rotlayer1, ansatz, rotlayer2)

    return QuantumCircuit(n_qubits, block)


@pytest.mark.parametrize("diff_mode", ["ad", "gpsr"])
def test_parametrized_rotation(diff_mode: str) -> None:
    param = Parameter("theta", trainable=False)
    nqubits = 2
    block1 = RX(0, param)
    block2 = Z(1)
    comp_block = chain(block1, block2)
    circ = QuantumCircuit(nqubits, comp_block)

    backend = backend_factory("pyqtorch", diff_mode=diff_mode)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)

    values = {param.name: torch.tensor([np.pi])}
    wf = backend.run(pyqtorch_circ, embed(params, values))[0]

    wf_prob = torch.abs(torch.pow(wf, 2))  # type: ignore [arg-type]
    assert torch.allclose(wf_prob.reshape(nqubits, nqubits), expected_pi)

    values = {param.name: torch.tensor([np.pi / 2])}
    wf = backend.run(pyqtorch_circ, embed(params, values))[0]
    wf_prob = torch.abs(torch.pow(wf, 2))
    assert torch.allclose(wf_prob.reshape(nqubits, nqubits), expected_pi2)


@pytest.mark.parametrize("diff_mode", ["ad", "gpsr"])
def test_parametrized_rotation_with_expr(diff_mode: str) -> None:
    param = Parameter("theta", trainable=False)
    nqubits = 2
    block1 = RX(0, sympy.exp(5 * param))
    block2 = Z(1)
    comp_block = chain(block1, block2)
    circ = QuantumCircuit(nqubits, comp_block)

    backend = backend_factory("pyqtorch", diff_mode=diff_mode)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)

    angle = np.log(np.pi) / 5
    values = {param.name: torch.tensor([angle])}
    wf = backend.run(pyqtorch_circ, embed(params, values))[0]
    wf_prob = torch.abs(torch.pow(wf, 2))  # type: ignore [arg-type]
    assert torch.allclose(wf_prob.reshape(nqubits, nqubits), expected_pi)

    angle = np.log(np.pi / 2) / 5
    values = {param.name: torch.tensor([angle])}
    wf = backend.run(pyqtorch_circ, embed(params, values))[0]
    wf_prob = torch.abs(torch.pow(wf, 2))  # type: ignore [arg-type]
    assert torch.allclose(wf_prob.reshape(nqubits, nqubits), expected_pi2)


def test_embeddings() -> None:
    n_qubits = 4
    circ = parametric_circuit(n_qubits)
    backend = backend_factory("pyqtorch", diff_mode="ad")
    (_, _, embed, params) = backend.convert(circ)

    batch_size = 5

    inputs = {"x": torch.ones(batch_size), "y": torch.rand(batch_size)}
    low_level_params = embed(params, inputs)

    assert len(list(low_level_params.keys())) == 9

    assert [v for k, v in low_level_params.items() if k.startswith("fix_")][0] == 0.5
    assert torch.allclose(low_level_params["3*x"], 3 * inputs["x"])
    assert torch.allclose(low_level_params["x"], inputs["x"])
    assert torch.allclose(low_level_params["exp(y)"], torch.exp(inputs["y"]))

    with pytest.raises(KeyError):
        embed(params, {"x": torch.ones(batch_size)})


@pytest.mark.parametrize("batch_size", [1, 5])
def test_expval_differentiation_all_diffmodes(batch_size: int) -> None:
    grads = {}
    for diff_mode in ["adjoint", "ad", "gpsr"]:
        n_qubits = 4
        observable: list[AbstractBlock] = [kron(Z(i) for i in range(n_qubits))]
        circ = parametric_circuit(n_qubits)

        bknd = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
        pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = bknd.convert(circ, observable)

        inputs_x = torch.rand(batch_size, requires_grad=True)
        inputs_y = torch.rand(batch_size, requires_grad=True)

        def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            inputs = {"x": x, "y": y}
            all_params = embeddings_fn(params, inputs)
            return bknd.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

        exp = func(inputs_x, inputs_y)
        grad = torch.autograd.grad(exp, inputs_x, torch.ones_like(exp))[0]
        grads[diff_mode] = grad.detach()

    for k, v in grads.items():
        assert torch.allclose(grads["ad"], v)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        pytest.param(
            "2",
            marks=pytest.mark.xfail(
                reason="Batch_size and n_obs > 1 should be made consistent."  # FIXME
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "diff_mode",
    [
        "ad",
        pytest.param(
            "gpsr",
            marks=pytest.mark.xfail(reason="PSR cannot be applied to parametric observable."),
        ),
    ],
)
def test_parametricobs_expval_differentiation(batch_size: int, diff_mode: str) -> None:
    torch.manual_seed(42)
    n_qubits = 4
    observable: list[AbstractBlock] = [add(Z(i) * Parameter(f"o_{i}") for i in range(n_qubits))]
    n_obs = len(observable)
    circ = parametric_circuit(n_qubits)

    ad_backend = backend_factory(backend="pyqtorch", diff_mode=diff_mode)
    pyqtorch_circ, pyqtorch_obs, embeddings_fn, params = ad_backend.convert(circ, observable)

    inputs_x = torch.rand(batch_size, requires_grad=True)
    inputs_y = torch.rand(batch_size, requires_grad=True)
    param_w = torch.rand(1, requires_grad=True)

    def func(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # FIXME: add a parameter from a parametric observable
        inputs = {"x": x, "y": y}
        params["o_1"] = w
        all_params = embeddings_fn(params, inputs)
        return ad_backend.expectation(pyqtorch_circ, pyqtorch_obs, all_params)

    expval = func(inputs_x, inputs_y, param_w)
    assert torch.autograd.gradcheck(func, (inputs_x, inputs_y, param_w))
    assert torch.autograd.gradgradcheck(func, (inputs_x, inputs_y, param_w))

    assert torch.allclose(
        finitediff(lambda x: func(x, inputs_y, param_w), inputs_x),
        torch.autograd.grad(expval, inputs_x, torch.ones_like(expval), create_graph=True)[0],
    )

    assert torch.allclose(
        finitediff(lambda w: func(inputs_x, inputs_y, w), param_w),
        torch.autograd.grad(expval, param_w, torch.ones_like(expval), create_graph=True)[0],
    )
