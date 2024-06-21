from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import qutip
import torch
from metrics import MIDDLE_ACCEPTANCE
from pyqtorch.utils import SolverType

from qadence import (
    AbstractBlock,
    HamEvo,
    QuantumCircuit,
    QuantumModel,
    Register,
)


@pytest.mark.parametrize("ode_solver", [SolverType.DP5_SE, SolverType.KRYLOV_SE])
def test_time_dependent_generator(
    qadence_generator: AbstractBlock,
    qutip_generator: Callable,
    feature_param_x: float,
    feature_param_y: float,
    ode_solver: SolverType,
) -> None:
    duration = 1.0
    n_steps = 500

    # simulate with qadence HamEvo
    hamevo = HamEvo(qadence_generator, 0.0, duration=duration)
    reg = Register(2)
    circ = QuantumCircuit(reg, hamevo)
    model = QuantumModel(circ, configuration={"ode_solver": ode_solver, "n_steps_hevo": n_steps})
    state_qadence = model.run(
        values={"x": torch.tensor(feature_param_x), "y": torch.tensor(feature_param_y)}
    )

    # simulate with qutip
    t_points = np.linspace(0, duration, n_steps)
    psi_0 = qutip.basis(4, 0)
    result = qutip.sesolve(qutip_generator, psi_0, t_points)
    state_qutip = torch.tensor(result.states[-1].full().T)

    assert torch.allclose(state_qadence, state_qutip, atol=MIDDLE_ACCEPTANCE)
