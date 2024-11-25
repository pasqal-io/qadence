from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import qutip
import torch
from metrics import MIDDLE_ACCEPTANCE
from pyqtorch.utils import SolverType

from qadence import AbstractBlock, HamEvo, QuantumCircuit, QuantumModel, Register, run


@pytest.mark.parametrize("duration", [0.5, 1.0])
@pytest.mark.parametrize("ode_solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_time_dependent_generator(
    qadence_generator: AbstractBlock,
    qutip_generator: Callable,
    time_param: str,
    feature_param_x: float,
    feature_param_y: float,
    ode_solver: SolverType,
    duration: float,
) -> None:
    n_steps = 500

    psi_0 = qutip.basis(4, 0)
    psi0_torch = (torch.tensor(psi_0.full()).to(torch.complex128)).reshape(1, -1)

    # simulate with qadence HamEvo using QuantumModel
    hamevo = HamEvo(qadence_generator, time_param)
    reg = Register(2)
    circ = QuantumCircuit(reg, hamevo)
    config = {"ode_solver": ode_solver, "n_steps_hevo": n_steps}
    values = {
        "x": torch.tensor(feature_param_x),
        "y": torch.tensor(feature_param_y),
        "duration": torch.tensor(duration),
    }

    model = QuantumModel(circ, configuration=config)
    state_qadence0 = model.run(values=values, state=psi0_torch)

    # simulate with qadence.execution
    state_qadence1 = run(hamevo, values=values, configuration=config, state=psi0_torch)

    # simulate with qutip
    t_points = np.linspace(0, duration, n_steps)

    result = qutip.sesolve(qutip_generator, psi_0, t_points)
    state_qutip = torch.tensor(result.states[-1].full().T)

    assert torch.allclose(state_qadence0, state_qutip, atol=MIDDLE_ACCEPTANCE)
    assert torch.allclose(state_qadence1, state_qutip, atol=MIDDLE_ACCEPTANCE)


@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_noisy_time_dependent_generator(
    qadence_generator: AbstractBlock,
    qutip_generator: Callable,
    time_param: str,
    feature_param_x: float,
    feature_param_y: float,
    duration: float,
) -> None:
    n_steps = 500
    ode_solver = SolverType.DP5_ME

    # initial state
    psi_0 = qutip.basis(4, 0)
    psi0_torch = (torch.tensor(psi_0.full()).to(torch.complex128)).reshape(1, -1)

    # Define jump operators
    # Note that we squeeze to remove the batch dimension
    list_ops = [torch.eye(4, dtype=torch.complex128)]

    # simulate with qadence HamEvo using QuantumModel
    hamevo = HamEvo(qadence_generator, time_param, noise_operators=list_ops)
    reg = Register(2)
    circ = QuantumCircuit(reg, hamevo)
    config = {"ode_solver": ode_solver, "n_steps_hevo": n_steps}
    values = {
        "x": torch.tensor(feature_param_x),
        "y": torch.tensor(feature_param_y),
        "duration": torch.tensor(duration),
    }

    model = QuantumModel(circ, configuration=config)
    state_qadence0 = model.run(values=values, state=psi0_torch)

    # simulate with qadence.execution
    state_qadence1 = run(hamevo, values=values, configuration=config, state=psi0_torch)

    # simulate with qutip
    t_points = np.linspace(0, duration, n_steps)

    # psi_0 = psi_0 * psi_0.dag()
    list_ops_qutip = [qutip.qeye(4)]
    result = qutip.mesolve(qutip_generator, psi_0, t_points, list_ops_qutip)
    state_qutip = torch.tensor(result.states[-1].full().T).unsqueeze(0)
    assert torch.allclose(state_qadence0, state_qutip, atol=MIDDLE_ACCEPTANCE)
    assert torch.allclose(state_qadence1, state_qutip, atol=MIDDLE_ACCEPTANCE)
