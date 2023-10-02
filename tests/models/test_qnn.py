from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from qadence import BackendName, DiffMode, FeatureParameter, QuantumCircuit
from qadence.blocks import (
    chain,
    kron,
    parameters,
    tag,
)
from qadence.constructors import hea, ising_hamiltonian, total_magnetization
from qadence.models import QNN
from qadence.operations import RX, RY
from qadence.parameters import Parameter
from qadence.states import uniform_state
from qadence.transpile import set_trainable


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
    qnn = QNN(circuit, observable)
    assert qnn.in_features == dim

    res: torch.Tensor = qnn(a)
    assert qnn.out_features is not None and qnn.out_features == 1
    assert len(res.size()) == qnn.out_features
    assert len(res) == batch_size


def test_qnn_expectation(n_qubits: int = 4) -> None:
    theta0 = Parameter("theta0", trainable=True)
    theta1 = Parameter("theta1", trainable=True)

    ry0 = RY(0, theta0)
    ry1 = RY(1, theta1)

    fm = chain(ry0, ry1)

    ansatz = hea(2, 2, param_prefix="eps")

    block = chain(fm, ansatz)

    qc = QuantumCircuit(n_qubits, block)
    uni_state = uniform_state(n_qubits)
    obs = total_magnetization(n_qubits)
    model = QNN(circuit=qc, observable=obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

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


def test_multiparam_qnn_training() -> None:
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
    qnn = QNN(qc, observable=obs, diff_mode=DiffMode.AD, backend=backend)

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
