from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from qadence import QNN
from qadence.blocks import (
    chain,
    kron,
    tag,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors import hamiltonian_factory, hea, ising_hamiltonian, total_magnetization
from qadence.operations import RX, RY, Z
from qadence.parameters import FeatureParameter, Parameter
from qadence.states import uniform_state
from qadence.types import BackendName, DiffMode


def build_circuit(n_qubits_per_feature: int, n_features: int, depth: int = 2) -> QuantumCircuit:
    n_qubits = n_qubits_per_feature * n_features

    idx_fms = []

    for i in range(n_features):
        start_qubit = i * n_qubits_per_feature
        end_qubit = (i + 1) * n_qubits_per_feature
        param = FeatureParameter(f"x{i}")
        block = kron(*[RY(qubit, (qubit + 1) * param) for qubit in range(start_qubit, end_qubit)])
        idx_fm = tag(block, tag=f"FM{i}")
        idx_fms.append(idx_fm)

    fm = kron(*idx_fms)
    ansatz = hea(n_qubits, depth=depth)

    return QuantumCircuit(n_qubits, fm, ansatz)


def test_parameters(parametric_circuit: QuantumCircuit) -> None:
    circ = parametric_circuit
    model = QNN(
        circ,
        observable=total_magnetization(circ.n_qubits),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )

    vparams = model.vparams
    assert isinstance(vparams, OrderedDict)

    trainables: list[Parameter]
    trainables = [p for p in circ.parameters() if not p.is_number and p.trainable]  # type: ignore
    assert model.num_vparams == len(trainables)

    # init with torch
    init_values_tc = torch.rand(model.num_vparams)
    model.reset_vparams(init_values_tc)  # type: ignore
    assert torch.equal(init_values_tc, model.vals_vparams)

    # init with numpy
    init_values_np = np.random.rand(model.num_vparams)
    model.reset_vparams(init_values_np)  # type: ignore
    assert torch.equal(torch.tensor(init_values_np), model.vals_vparams)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_input_nd(dim: int) -> None:
    batch_size = 10
    n_qubits_per_feature = 2

    observable = total_magnetization(n_qubits_per_feature * dim)
    circuit = build_circuit(n_qubits_per_feature, dim)
    a = torch.rand(batch_size, dim)
    qnn = QNN(circuit, observable, inputs=[f"x{i}" for i in range(dim)])
    assert qnn.in_features == dim

    res: torch.Tensor = qnn(a)
    assert qnn.out_features is not None and qnn.out_features == 1
    assert res.size()[1] == qnn.out_features
    assert res.size()[0] == batch_size


@pytest.mark.parametrize("diff_mode", ["ad", "adjoint"])
def test_qnn_expectation(diff_mode: str, n_qubits: int = 2) -> None:
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)

    ry0 = RY(0, theta0)
    ry1 = RY(1, theta1)

    fm = chain(ry0, ry1)

    ansatz = hea(n_qubits, depth=2, param_prefix="eps")

    block = chain(fm, ansatz)

    qc = QuantumCircuit(n_qubits, block)
    uni_state = uniform_state(n_qubits)
    obs = total_magnetization(n_qubits)
    model = QNN(circuit=qc, observable=obs, backend=BackendName.PYQTORCH, diff_mode=diff_mode)

    exp = model(values={}, state=uni_state)
    assert not torch.any(torch.isnan(exp))


def test_qnn_multiple_outputs(n_qubits: int = 4) -> None:
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)
    phi = Parameter("phi", trainable=False)

    ry_theta0 = RY(0, theta0)
    ry_theta1 = RY(1, theta1)

    fm = chain(ry_theta0, ry_theta1, *[RX(i, phi) for i in range(n_qubits)])
    ansatz = hea(2, 2, param_prefix="eps")
    block = chain(fm, ansatz)

    qc = QuantumCircuit(n_qubits, block)
    uni_state = uniform_state(n_qubits)

    obs = []
    n_obs = 3
    for i in range(n_obs):
        o = float(i + 1) * ising_hamiltonian(4)
        obs.append(o)

    model = QNN(circuit=qc, observable=obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    assert model.out_features == n_obs
    assert len(model._observable) == n_obs  # type: ignore[arg-type]

    batch_size = 10
    values = {"phi": torch.rand(batch_size)}
    exp = model(values=values, state=uni_state)
    assert not torch.any(torch.isnan(exp))
    assert exp.shape[0] == batch_size and exp.shape[1] == n_obs

    factors = torch.linspace(1, n_obs, n_obs)
    for i, e in enumerate(exp):
        tmp = torch.div(e, factors * e[0])
        assert torch.allclose(tmp, torch.ones(n_obs))


@pytest.mark.parametrize("diff_mode", ["ad", "adjoint"])
def test_multiparam_qnn_training(diff_mode: str) -> None:
    backend = BackendName.PYQTORCH
    n_qubits = 2
    n_epochs = 5

    x = Parameter("x", trainable=False)
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)

    ry0 = RY(0, theta0 * x)
    ry1 = RY(1, theta1 * x)

    fm = chain(ry0, ry1)

    ansatz = hea(n_qubits, depth=2, param_prefix="eps")

    block = chain(fm, ansatz)
    qc = QuantumCircuit(n_qubits, block)
    obs = total_magnetization(n_qubits)
    qnn = QNN(qc, observable=obs, diff_mode=diff_mode, backend=backend)

    optimizer = torch.optim.Adam(qnn.parameters(), lr=1e-1)

    loss_fn = torch.nn.MSELoss()
    for i in range(n_epochs):
        optimizer.zero_grad()
        exp = qnn(values={"x": 1.0}, state=None)
        assert not torch.any(torch.isnan(exp))
        loss = loss_fn(exp, torch.tensor([np.random.rand()], requires_grad=False))
        assert not torch.any(torch.isnan(loss))
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1} modeling training - Loss: {loss.item()}")


def test_qnn_input_order() -> None:
    from torch import cos, sin

    def compute_state_manually(xs: torch.Tensor) -> torch.Tensor:
        x, y = xs[0], xs[1]
        return torch.tensor(
            [
                cos(0.5 * y) * cos(0.5 * x),
                -1j * cos(0.5 * x) * sin(0.5 * y),
                -1j * cos(0.5 * y) * sin(0.5 * x),
                -sin(0.5 * x) * sin(0.5 * y),
            ]
        )

    xs = torch.rand(5, 2)
    ys = torch.vstack(list(map(compute_state_manually, xs)))

    model = QNN(
        QuantumCircuit(
            2,
            chain(
                RX(0, FeatureParameter("x")),
                RX(1, FeatureParameter("y")),
            ),
        ),
        observable=total_magnetization(2),
        inputs=["x", "y"],
    )
    assert torch.allclose(ys, model.run(xs))

    # now try again with switched featuremap order
    model = QNN(
        QuantumCircuit(
            2,
            chain(
                RX(1, FeatureParameter("y")),
                RX(0, FeatureParameter("x")),
            ),
        ),
        observable=total_magnetization(2),
        inputs=["x", "y"],
    )
    assert torch.allclose(ys, model.run(xs))

    # make sure it fails with wrong order
    model = QNN(
        QuantumCircuit(
            2,
            chain(
                RX(1, FeatureParameter("y")),
                RX(0, FeatureParameter("x")),
            ),
        ),
        observable=total_magnetization(2),
        inputs=["y", "x"],
    )
    assert not torch.allclose(ys, model.run(xs))


def quantum_circuit(n_qubits: int = 2, depth: int = 1) -> QuantumCircuit:
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    tag(fm, "feature_map")

    ansatz = hea(n_qubits=n_qubits, depth=depth)
    tag(ansatz, "ansatz")

    return QuantumCircuit(n_qubits, fm, ansatz)


def get_qnn(
    n_qubits: int,
    depth: int,
    inputs: list = None,
) -> QNN:
    observable = hamiltonian_factory(n_qubits, detuning=Z)
    circuit = quantum_circuit(n_qubits=n_qubits, depth=depth)
    model = QNN(
        circuit,
        observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
        inputs=inputs,
    )
    return model


@pytest.mark.parametrize("output_scale", [1.0, 2.0])
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_transformed_module(output_scale: float, batch_size: int, n_qubits: int) -> None:
    depth = 1
    fparam = "phi"
    input_values = {fparam: torch.rand(batch_size, requires_grad=True)}
    model = get_qnn(n_qubits, depth, inputs=[fparam])
    transformed_model = get_qnn(
        n_qubits,
        depth,
        inputs=[fparam],
    )
    init_params = torch.rand(model.num_vparams)
    model.reset_vparams(init_params)
    transformed_model.reset_vparams(init_params)
    pred = model(input_values)
    transformed_pred = transformed_model(input_values)
    assert torch.allclose(output_scale * pred, transformed_pred)
