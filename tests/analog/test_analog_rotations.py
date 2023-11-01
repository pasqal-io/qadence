from __future__ import annotations

import pytest
import torch
from metrics import ATOL_DICT  # type: ignore

from qadence import BackendName, Register, add_interaction
from qadence.blocks import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.models import QuantumModel
from qadence.operations import AnalogRX, AnalogRY, AnalogRZ
from qadence.parameters import FeatureParameter
from qadence.states import equivalent_state, random_state
from qadence.types import DiffMode


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("spacing", [4.0, 8.0, 15.0])
@pytest.mark.parametrize("rot_op", [AnalogRX, AnalogRY, AnalogRZ])
def test_analog_rxyz_run(n_qubits: int, spacing: float, rot_op: AbstractBlock) -> None:
    init_state = random_state(n_qubits)

    phi = FeatureParameter("phi")

    block = rot_op(phi)  # type: ignore [operator]

    register = Register.line(n_qubits)
    circuit = QuantumCircuit(register, block)

    circuit_pyqtorch = add_interaction(circuit, spacing=spacing)

    model_pyqtorch = QuantumModel(circuit_pyqtorch, backend=BackendName.PYQTORCH)

    conf = {"spacing": spacing}

    model_pulser = QuantumModel(
        circuit,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        configuration=conf,
    )

    batch_size = 5
    values = {"phi": 1.0 + torch.rand(batch_size)}

    wf_pyq = model_pyqtorch.run(values=values, state=init_state)
    wf_pulser = model_pulser.run(values=values, state=init_state)

    assert equivalent_state(wf_pyq, wf_pulser, atol=ATOL_DICT[BackendName.PULSER])
