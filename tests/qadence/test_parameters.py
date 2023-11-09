from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch
from torch import allclose

from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import ParametricBlock, chain
from qadence.blocks.utils import expressions
from qadence.circuit import QuantumCircuit
from qadence.constructors import hea, total_magnetization
from qadence.models import QuantumModel
from qadence.operations import CNOT, RX, RY, RZ
from qadence.parameters import (
    FeatureParameter,
    Parameter,
    evaluate,
    stringify,
)
from qadence.serialization import deserialize, serialize
from qadence.states import one_state, uniform_state, zero_state
from qadence.types import BackendName, DiffMode


def test_param_initialization(parametric_circuit: QuantumCircuit) -> None:
    circ = parametric_circuit

    # check general configuration
    assert len(circ.unique_parameters) == 4
    # the additional 4 parameters are the four fixed scale parameters of the observable
    assert len(circ.parameters()) == 6

    # unique parameters are returned as sympy symbols
    for p in circ.unique_parameters:
        if p is not None:
            assert isinstance(p, sympy.Symbol)

    params: list[Parameter]
    params = circ.parameters()  # type: ignore [assignment]
    assert all([isinstance(p, sympy.Basic) for p in params])

    # check symbol assignation
    non_number = [p for p in params if not p.is_number]
    expected = ["x", "theta1", "theta2", "theta3"]
    # symbols = unique_symbols(params)
    assert len(non_number) == len(expected)
    assert all([a in expected for a in non_number])

    # check numerical valued parameter
    for q in params[:6]:
        if q.is_number:
            assert evaluate(q) == np.pi
    for q in params[6:]:
        assert evaluate(q) == 1.0

    # check parameter with expression
    exprs = expressions(circ.block)
    for expr in exprs:
        if not expr.is_number and "x" in stringify(expr):
            assert stringify(expr) == "3*x"


@pytest.mark.parametrize(
    "n_qubits",
    [1, 2, 4, 6, 8],
)
def test_multiparam_expressions(n_qubits: int) -> None:
    w = Parameter("w", trainable=True)
    x = Parameter("x", trainable=True)
    y = Parameter("y", trainable=True)
    z = Parameter("z", trainable=True)
    block = RX(np.random.randint(n_qubits), w * x)
    block1 = RZ(np.random.randint(n_qubits), y + z)
    qc = QuantumCircuit(n_qubits, chain(block, block1))
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, BackendName.PYQTORCH, DiffMode.AD)
    uni_state = uniform_state(n_qubits)
    wf = qm.run(
        {
            "w": torch.rand(1) * np.pi,
            "x": torch.rand(1) * np.pi,
            "y": torch.rand(1) * np.pi,
            "z": torch.rand(1) * np.pi,
        },
        uni_state,
    )
    assert wf is not None


def test_multiparam_no_rx_rotation(n_qubits: int = 1) -> None:
    w = Parameter("w", trainable=True, value=0.0)
    x = Parameter("x", trainable=True, value=0.0)
    y = Parameter("y", trainable=True, value=0.0)
    block = RX(np.random.randint(n_qubits), x + y * w)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, BackendName.PYQTORCH, DiffMode.AD)
    uni_state = uniform_state(n_qubits)
    wf = qm.run(
        {},
        uni_state,
    )

    assert allclose(wf, uni_state)


def test_multiparam_pi_ry_rotation_trainable(n_qubits: int = 1) -> None:
    x = Parameter("x", trainable=True, value=torch.tensor([np.pi / 2], dtype=torch.cdouble))
    y = Parameter("y", trainable=True, value=torch.tensor([np.pi / 2], dtype=torch.cdouble))
    block = RY(0, x + y)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    o_state = one_state(n_qubits)
    wf = qm.run({}, z_state)
    assert torch.allclose(wf, o_state)


def test_multiparam_pi_ry_rotation_nontrainable(n_qubits: int = 1) -> None:
    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=False)
    block = RY(0, x + y)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    o_state = one_state(n_qubits)
    wf = qm.run(
        {
            "x": torch.tensor([np.pi / 2], dtype=torch.cdouble),
            "y": torch.tensor([np.pi / 2], dtype=torch.cdouble),
        },
        z_state,
    )
    assert torch.allclose(wf, o_state)


def test_mixed_single_trainable(n_qubits: int = 1) -> None:
    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=True, value=torch.tensor([np.pi / 2], dtype=torch.cdouble))
    ry0 = RY(0, x)
    ry1 = RY(0, y)
    block = chain(ry0, ry1)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    o_state = one_state(n_qubits)
    wf = qm.run(
        {
            "x": torch.tensor([np.pi / 2], dtype=torch.cdouble),
        },
        z_state,
    )
    assert torch.allclose(wf, o_state)


def test_multiple_trainable_multiple_untrainable(n_qubits: int = 1) -> None:
    w = Parameter("w", trainable=True)
    x = Parameter("x", trainable=True)
    y = Parameter("y", trainable=True)
    rx = RY(0, x + y * w)

    a = Parameter("a", trainable=False)
    b = Parameter("b", trainable=False)
    c = Parameter("c", trainable=False)
    rz = RZ(0, a - b - c)

    block = chain(rx, rz)

    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, BackendName.PYQTORCH, DiffMode.AD)
    uni_state = uniform_state(n_qubits)
    wf = qm.run(
        {param: np.random.rand() for param in ["a", "b", "c"]},
        uni_state,
    )

    assert not torch.any(torch.isnan(wf))


def test_multparam_grads(n_qubits: int = 2) -> None:
    batch_size = 5
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)
    phi = Parameter("phi", trainable=False)

    variational = RY(1, theta0 * theta1)
    fm = RX(0, phi)
    block = chain(fm, variational, CNOT(0, 1))

    circ = QuantumCircuit(n_qubits, block)

    # Making circuit with AD
    observable = total_magnetization(n_qubits=n_qubits)
    quantum_backend = PyQBackend()
    (pyq_circ, pyq_obs, embed, params) = quantum_backend.convert(circ, observable)

    batch_size = 5
    values = {
        "phi": torch.rand(batch_size, requires_grad=False),
    }

    wf = quantum_backend.run(pyq_circ, embed(params, values))
    expval = quantum_backend.expectation(pyq_circ, pyq_obs, embed(params, values))
    dexpval_x = torch.autograd.grad(
        expval, params["theta0"], torch.ones_like(expval), retain_graph=True
    )[0]
    dexpval_y = torch.autograd.grad(
        expval, params["theta1"], torch.ones_like(expval), retain_graph=True
    )[0]
    assert (
        not torch.isnan(wf).any().item()
        and not torch.isnan(dexpval_x).any().item()
        and not torch.isnan(dexpval_y).any().item()
    )


def test_non_trainable_trainable_gate(n_qubits: int = 1) -> None:
    x = Parameter("x", trainable=True, value=torch.tensor([1.0], dtype=torch.cdouble))
    y = Parameter("y", trainable=False)
    z = Parameter(
        "z",
        trainable=True,
        value=torch.tensor([np.pi / 2], dtype=torch.cdouble),
    )
    block = RY(0, x * y + z)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    o_state = one_state(n_qubits)
    wf = qm.run(
        {
            "y": torch.tensor([np.pi / 2], dtype=torch.cdouble),
        },
        z_state,
    )
    assert torch.allclose(wf, o_state)


def test_trainable_untrainable_fm(n_qubits: int = 2) -> None:
    x = Parameter("x", trainable=False)
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)

    ry0 = RY(0, theta0 * x)
    ry1 = RY(1, theta1 * x)

    fm = chain(ry0, ry1)

    ansatz = hea(2, 2, param_prefix="eps")

    block = chain(fm, ansatz)

    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    wf = qm.run(
        {
            "x": torch.tensor([1.0], dtype=torch.cdouble),
            "theta0": torch.tensor([np.pi / 2], dtype=torch.cdouble),
            "theta1": torch.tensor([np.pi / 2], dtype=torch.cdouble),
        },
        z_state,
    )
    assert wf is not None


def test_hetereogenous_multiparam_expr(n_qubits: int = 2) -> None:
    x = Parameter("x", trainable=False)
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)
    myconstant = 2.0

    ry0 = RY(0, theta0 * x + myconstant)
    ry1 = RY(1, theta1 * x - myconstant)

    fm = chain(ry0, ry1)

    ansatz = hea(2, 2, param_prefix="eps")

    block = chain(fm, ansatz)

    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qm = QuantumModel(qc, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    z_state = zero_state(n_qubits)
    wf = qm.run(
        {
            "x": torch.tensor([1.0], dtype=torch.cdouble),
            "theta0": torch.tensor([np.pi / 2], dtype=torch.cdouble),
            "theta1": torch.tensor([np.pi / 2], dtype=torch.cdouble),
        },
        z_state,
    )
    assert wf is not None


def test_single_param_serialization() -> None:
    x0 = Parameter("x", trainable=True, value=1.0)
    d0 = x0._to_dict()
    x1 = Parameter._from_dict(d0)
    assert x0 == x1

    y = Parameter("y", trainable=True)
    d2 = y._to_dict()
    y1 = Parameter._from_dict(d2)
    assert y == y1


@pytest.mark.parametrize(
    "gate",
    [
        RX(0, "theta"),
        RY(0, Parameter("theta", trainable=False)),
        RZ(0, Parameter("theta", trainable=True, value=5.0)),
    ],
)
def test_serialize_singleparam_gate(gate: ParametricBlock) -> None:
    d = serialize(gate.parameters.parameter)
    op = deserialize(d)
    assert gate.parameters.parameter == op


def test_multiparam_serialization() -> None:
    x = Parameter("x", trainable=True, value=1.0)
    y = Parameter("y", trainable=True, value=2.0)
    expr = x + y
    myrx = RX(0, expr)
    d_block = myrx._to_dict()
    nb = RX._from_dict(d_block)
    assert nb == myrx


def test_multiparam_eval_serialization() -> None:
    x = Parameter("x", trainable=True, value=1.0)
    y = Parameter("y", trainable=True, value=2.0)
    expr = x + y
    myrx = RX(0, expr)
    d = serialize(myrx.parameters.parameter)
    loaded_expr = deserialize(d)
    assert loaded_expr == expr
    eval_orig = evaluate(myrx.parameters.parameter)
    eval_copy = evaluate(loaded_expr)
    assert eval_orig == eval_copy


def test_sympy_modules() -> None:
    x = FeatureParameter("x")
    y = FeatureParameter("y")
    expr = 2 * sympy.acos(x) + (sympy.cos(y) + sympy.asinh(y))
    d = serialize(expr)
    loaded_expr = deserialize(d)
    assert loaded_expr == expr
    assert evaluate(expr) == evaluate(loaded_expr)
