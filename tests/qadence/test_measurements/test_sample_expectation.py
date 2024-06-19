from __future__ import annotations

import pytest
import torch
from metrics import MEASUREMENT_ATOL
from torch import Tensor

from qadence import (
    AbstractBlock,
    QuantumCircuit,
    QuantumModel,
    add,
    chain,
    kron,
)
from qadence.measurements import Measurements
from qadence.measurements.samples import compute_expectation
from qadence.operations import CNOT, RX, Z
from qadence.types import BackendName


@pytest.mark.parametrize(
    "n_shots, block, observable, backend",
    [
        (
            10000,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 3)), CNOT(0, 1)),
            [add(kron(Z(0), Z(1)) + Z(0))],
            BackendName.PYQTORCH,
        ),
        (
            10000,
            chain(kron(RX(0, torch.pi / 4), RX(1, torch.pi / 5)), CNOT(0, 1)),
            [2 * Z(1) + 3 * Z(0), 3 * kron(Z(0), Z(1)) - 1 * Z(0)],
            BackendName.PYQTORCH,
        ),
        (
            10000,
            chain(kron(RX(0, torch.pi / 3), RX(1, torch.pi / 6)), CNOT(0, 1)),
            [add(Z(1), -Z(0)), 3 * kron(Z(0), Z(1)) + 2 * Z(0)],
            BackendName.PYQTORCH,
        ),
        (
            10000,
            chain(kron(RX(0, torch.pi / 6), RX(1, torch.pi / 4)), CNOT(0, 1)),
            [add(Z(1), -2 * Z(0)), add(2 * kron(Z(0), Z(1)), 4 * Z(0))],
            BackendName.PYQTORCH,
        ),
    ],
)
def test_sample_expectations(
    n_shots: int,
    block: AbstractBlock,
    observable: list[AbstractBlock],
    backend: BackendName,
) -> None:
    circuit = QuantumCircuit(block.n_qubits, block)
    tomo_measurement = Measurements(
        protocol=Measurements.TOMOGRAPHY,
        options={"n_shots": n_shots},
    )

    model = QuantumModel(
        circuit=circuit, observable=observable, measurement=tomo_measurement, backend=backend
    )
    expectation_tomo = model.expectation(measurement=tomo_measurement)[0]
    expectation_sampling = Tensor(compute_expectation(observable, model.sample()))

    assert torch.allclose(expectation_tomo, expectation_sampling, atol=MEASUREMENT_ATOL)
