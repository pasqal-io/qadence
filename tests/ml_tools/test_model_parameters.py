from __future__ import annotations

import torch

from qadence import BackendName, DiffMode, QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization
from qadence.ml_tools.parameters import get_parameters, num_parameters, set_parameters
from qadence.models import QNN


def test_get_parameters(Basic: torch.nn.Module) -> None:
    # verify that parameters have expected length
    model = Basic
    ps = get_parameters(model)
    assert len(ps) == model.n_neurons * 4 * 2 + 2 * 3
    assert len(ps) == num_parameters(model)


def test_get_parameters_qnn() -> None:
    # verify that parameters have expected length (exluding fix/non-trainable params)
    n_qubits, depth = 2, 4
    fm = feature_map(n_qubits)
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs = total_magnetization(n_qubits)

    # initialize and use the model
    model = QNN(circuit, obs, diff_mode=DiffMode.AD, backend=BackendName.PYQTORCH)
    ps = get_parameters(model)
    assert len(ps) == 6 * 4


def test_set_parameters_qnn() -> None:
    # make sure that only variational parameters are set
    n_qubits, depth = 2, 4
    fm = feature_map(n_qubits)
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs = total_magnetization(n_qubits)

    # initialize and use the model
    model = QNN(circuit, obs, diff_mode=DiffMode.AD, backend=BackendName.PYQTORCH)
    set_parameters(model, torch.rand(6 * 4))
