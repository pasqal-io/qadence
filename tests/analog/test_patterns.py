from __future__ import annotations

import pytest
import torch
from metrics import ATOL_DICT

from qadence import (
    AnalogRX,
    BackendName,
    DifferentiableBackend,
    DiffMode,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    total_magnetization,
)
from qadence.analog.addressing import AddressingPattern
from qadence.analog.interaction import add_interaction
from qadence.backends.pulser.backend import Backend as PulserBackend
from qadence.backends.pulser.config import Configuration
from qadence.backends.pyqtorch.backend import Backend as PyqBackend


@pytest.mark.parametrize(
    "amp,det",
    [(0.0, 10.0), (15.0, 0.0), (15.0, 9.0)],
)
@pytest.mark.parametrize(
    "spacing",
    [8.0, 30.0],
)
def test_pulser_pyq_addressing(amp: float, det: float, spacing: float) -> None:
    n_qubits = 3
    block = AnalogRX("x")
    circ = QuantumCircuit(n_qubits, block)

    # define addressing patterns
    rand_weights_amp = torch.rand(n_qubits)
    rand_weights_amp = rand_weights_amp / rand_weights_amp.sum()
    w_amp = {i: rand_weights_amp[i] for i in range(n_qubits)}
    rand_weights_det = torch.rand(n_qubits)
    rand_weights_det = rand_weights_det / rand_weights_det.sum()
    w_det = {i: rand_weights_det[i] for i in range(n_qubits)}
    p = AddressingPattern(
        n_qubits=n_qubits,
        det=det,
        amp=amp,
        weights_det=w_det,
        weights_amp=w_amp,
    )

    values = {"x": torch.linspace(0.5, 2 * torch.pi, 50)}
    obs = total_magnetization(n_qubits)
    conf = Configuration(addressing_pattern=p, spacing=spacing)

    # define pulser backend
    pulser_backend = PulserBackend(config=conf)  # type: ignore[arg-type]
    conv = pulser_backend.convert(circ, obs)
    pulser_circ, pulser_obs, embedding_fn, params = conv
    diff_backend = DifferentiableBackend(pulser_backend, diff_mode=DiffMode.GPSR)
    expval_pulser = diff_backend.expectation(pulser_circ, pulser_obs, embedding_fn(params, values))

    # define pyq backend
    int_circ = add_interaction(circ, spacing=spacing, pattern=p)
    pyq_backend = PyqBackend()  # type: ignore[arg-type]
    conv = pyq_backend.convert(int_circ, obs)
    pyq_circ, pyq_obs, embedding_fn, params = conv
    diff_backend = DifferentiableBackend(pyq_backend, diff_mode=DiffMode.AD)
    expval_pyq = diff_backend.expectation(pyq_circ, pyq_obs, embedding_fn(params, values))

    torch.allclose(expval_pulser, expval_pyq, atol=ATOL_DICT[BackendName.PULSER])


@pytest.mark.flaky(max_runs=10)
def test_addressing_training() -> None:
    n_qubits = 3
    spacing = 8
    f_value = torch.rand(1)

    # define training parameters
    w_amp = {i: Parameter(f"w_amp{i}", trainable=True) for i in range(n_qubits)}
    w_det = {i: Parameter(f"w_det{i}", trainable=True) for i in range(n_qubits)}
    amp = Parameter("amp", trainable=True)
    det = Parameter("det", trainable=True)
    p = AddressingPattern(
        n_qubits=n_qubits,
        det=det,
        amp=amp,
        weights_det=w_det,  # type: ignore [arg-type]
        weights_amp=w_amp,  # type: ignore [arg-type]
    )

    # define training circuit
    circ = QuantumCircuit(n_qubits, AnalogRX(1 + torch.rand(1).item()))
    circ = add_interaction(circ, spacing=spacing, pattern=p)

    # define quantum model
    obs = total_magnetization(n_qubits)
    model = QuantumModel(circuit=circ, observable=obs)

    # prepare for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)
    loss_criterion = torch.nn.MSELoss()
    n_epochs = 100
    loss_save = []

    # train model
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model.expectation({})
        loss = loss_criterion(f_value, out)
        loss.backward()
        optimizer.step()
        loss_save.append(loss.item())

    # get final results
    f_value_model = model.expectation({}).detach()

    weights_amp = torch.tensor(list(p.evaluate(p.weights_amp, model.vparams).values()))
    weights_amp_mask = weights_amp.abs() < 0.001
    weights_amp[weights_amp_mask] = 0.0

    weights_det = torch.tensor(list(p.evaluate(p.weights_det, model.vparams).values()))
    weights_det_mask = weights_det.abs() < 0.001
    weights_det[weights_det_mask] = 0.0

    assert torch.all(weights_amp >= 0.0) and torch.all(weights_amp <= 1.0)
    assert torch.all(weights_det >= 0.0) and torch.all(weights_det <= 1.0)
    assert torch.isclose(f_value, f_value_model, atol=ATOL_DICT[BackendName.PULSER])
