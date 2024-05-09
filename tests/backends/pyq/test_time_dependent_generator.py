from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import qutip
import torch
from metrics import MIDDLE_ACCEPTANCE

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
def feature_param_value() -> float:
    return 2.0


@pytest.fixture
def qadence_generator(omega: float) -> AbstractBlock:
    t = TimeParameter("t")
    x = Parameter("x", trainable=False)
    generator_t = omega * (t * X(0) + x * (t**2) * Y(1))
    return generator_t  # type: ignore [no-any-return]


@pytest.fixture
def qutip_generator(omega: float, feature_param_value: float) -> Callable:
    def generator_t(t: float, args: Any) -> qutip.Qobj:
        return omega * (
            t * qutip.tensor(qutip.sigmax(), qutip.qeye(2))
            + feature_param_value * t**2 * qutip.tensor(qutip.qeye(2), qutip.sigmay())
        )

    return generator_t


def test_time_dependent_generator(
    qadence_generator: AbstractBlock, qutip_generator: Callable, feature_param_value: float
) -> None:
    duration = 1000  # ns

    # simulate with qadence HamEvo
    hamevo = HamEvo(qadence_generator, 0.0, duration=duration)
    reg = Register(2)
    circ = QuantumCircuit(reg, hamevo)
    model = QuantumModel(circ)
    state_qadence = model.run(values={"x": torch.tensor(feature_param_value)})

    # simulate with qutip
    t_points = np.linspace(0, duration, duration + 1) / 1000
    psi_0 = qutip.basis(4, 0)
    result = qutip.sesolve(qutip_generator, psi_0, t_points)
    state_qutip = torch.tensor(result.states[-1].full().T)

    assert torch.allclose(state_qadence, state_qutip, atol=MIDDLE_ACCEPTANCE)
