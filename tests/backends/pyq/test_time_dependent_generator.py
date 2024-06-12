from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import qutip
import sympy
import torch
from metrics import MIDDLE_ACCEPTANCE
from pasqal_solvers.utils import SolverType

from qadence import (
    AbstractBlock,
    HamEvo,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    Register,
    TimeParameter,
    X,
    Y,
)


@pytest.fixture
def omega() -> float:
    return 20.0


@pytest.fixture
def feature_param_x() -> float:
    return 2.0


@pytest.fixture
def feature_param_y() -> float:
    return 3.5


@pytest.fixture
def qadence_generator(omega: float) -> AbstractBlock:
    t = TimeParameter("t")
    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=False)
    generator_t = omega * (y * sympy.sin(t) * X(0) + x * (t**2) * Y(1))
    return generator_t  # type: ignore [no-any-return]


@pytest.fixture
def qutip_generator(omega: float, feature_param_x: float, feature_param_y: float) -> Callable:
    def generator_t(t: float, args: Any) -> qutip.Qobj:
        return omega * (
            feature_param_y * np.sin(t) * qutip.tensor(qutip.sigmax(), qutip.qeye(2))
            + feature_param_x * t**2 * qutip.tensor(qutip.qeye(2), qutip.sigmay())
        )

    return generator_t


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
