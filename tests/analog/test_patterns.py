from __future__ import annotations

import pytest
import torch
from metrics import LOW_ACCEPTANCE, MIDDLE_ACCEPTANCE

from qadence import (
    AnalogRX,
    AnalogRY,
    BackendName,
    DiffMode,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    Register,
    chain,
    total_magnetization,
)
from qadence.analog import AddressingPattern, IdealDevice
from qadence.states import equivalent_state


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
    x = Parameter("x")
    block = chain(AnalogRX(3 * x), AnalogRY(0.5 * x))

    # define addressing patterns
    rand_weights_amp = torch.rand(n_qubits)
    rand_weights_amp = rand_weights_amp / rand_weights_amp.sum()
    w_amp = {i: rand_weights_amp[i] for i in range(n_qubits)}
    rand_weights_det = torch.rand(n_qubits)
    rand_weights_det = rand_weights_det / rand_weights_det.sum()
    w_det = {i: rand_weights_det[i] for i in range(n_qubits)}

    pattern = AddressingPattern(
        n_qubits=n_qubits,
        det=det,
        amp=amp,
        weights_det=w_det,
        weights_amp=w_amp,
    )

    # define device specs
    device_specs = IdealDevice(pattern=pattern)

    reg = Register(support=n_qubits, spacing=spacing, device_specs=device_specs)
    circ = QuantumCircuit(reg, block)

    values = {"x": torch.linspace(0.5, 2 * torch.pi, 5)}
    obs = total_magnetization(n_qubits)

    # define pulser backend
    model = QuantumModel(
        circuit=circ,
        observable=obs,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
    )
    wf_pulser = model.run(values=values)
    expval_pulser = model.expectation(values=values)

    # define pyq backend
    model = QuantumModel(
        circuit=circ, observable=obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )
    wf_pyq = model.run(values=values)
    expval_pyq = model.expectation(values=values)

    assert equivalent_state(wf_pulser, wf_pyq, atol=MIDDLE_ACCEPTANCE)
    assert torch.allclose(expval_pulser, expval_pyq, atol=MIDDLE_ACCEPTANCE)


@pytest.mark.flaky(max_runs=5)
def test_addressing_training() -> None:
    n_qubits = 3
    f_value = torch.rand(1)

    # define training parameters
    w_amp = {i: f"w_amp{i}" for i in range(n_qubits)}
    w_det = {i: f"w_det{i}" for i in range(n_qubits)}
    amp = "amp"
    det = "det"

    # define pattern and device specs
    pattern = AddressingPattern(
        n_qubits=n_qubits,
        det=det,
        amp=amp,
        weights_det=w_det,  # type: ignore [arg-type]
        weights_amp=w_amp,  # type: ignore [arg-type]
    )

    device_specs = IdealDevice(pattern=pattern)

    reg = Register.line(n_qubits, spacing=8.0, device_specs=device_specs)

    # some otherwise fixed circuit
    block = AnalogRX(torch.pi)
    circ = QuantumCircuit(reg, block)

    # define quantum model
    obs = total_magnetization(n_qubits)
    model = QuantumModel(circuit=circ, observable=obs, backend=BackendName.PYQTORCH)

    # prepare for training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_criterion = torch.nn.MSELoss()
    n_epochs = 100
    loss_save = []

    # train model
    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = model.expectation({}).flatten()
        loss = loss_criterion(f_value, out)
        loss.backward()
        optimizer.step()
        loss_save.append(loss.item())

    # get final results
    f_value_model = model.expectation({}).detach()

    weights_amp = torch.tensor(list(pattern.evaluate(pattern.weights_amp, model.vparams).values()))
    weights_amp_mask = weights_amp.abs() < 0.001
    weights_amp[weights_amp_mask] = 0.0

    weights_det = torch.tensor(list(pattern.evaluate(pattern.weights_det, model.vparams).values()))
    weights_det_mask = weights_det.abs() < 0.001
    weights_det[weights_det_mask] = 0.0

    assert torch.all(weights_amp >= 0.0) and torch.all(weights_amp <= 1.0)
    assert torch.all(weights_det >= 0.0) and torch.all(weights_det <= 1.0)
    assert torch.isclose(f_value, f_value_model, atol=LOW_ACCEPTANCE)
