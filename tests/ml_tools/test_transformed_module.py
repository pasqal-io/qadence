from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from torch.nn import Parameter as TorchParam

from qadence import (
    BackendName,
    DiffMode,
    Parameter,
    QuantumCircuit,
)
from qadence.blocks import chain, tag
from qadence.constructors import hamiltonian_factory, hea
from qadence.models import QNN, transform_input, transform_output
from qadence.operations import RY, Z

np.random.seed(42)
torch.manual_seed(42)


def quantum_circuit(n_qubits: int = 2, depth: int = 1) -> QuantumCircuit:
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    tag(fm, "feature_map")

    ansatz = hea(n_qubits=n_qubits, depth=depth)
    tag(ansatz, "ansatz")

    return QuantumCircuit(n_qubits, fm, ansatz)


def get_qnn(
    n_qubits: int,
    depth: int,
    inputs: list = None,
    input_transform: Callable = lambda x: x,
    output_transform: Callable = lambda x: x,
) -> QNN:
    observable = hamiltonian_factory(n_qubits, detuning=Z)
    circuit = quantum_circuit(n_qubits=n_qubits, depth=depth)
    model = QNN(
        circuit,
        observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
        inputs=inputs,
        input_transform=input_transform,
        output_transform=output_transform,
    )
    return model


@pytest.mark.parametrize("output_scale", [1.0, 2.0])
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_transformed_module(output_scale: float, batch_size: int, n_qubits: int) -> None:
    depth = 1
    fparam = "phi"
    transform_input_fn = transform_input(
        input_scaling=TorchParam(torch.ones(batch_size)),
        input_shifting=torch.zeros(batch_size),
        inputs=[fparam],
    )
    transform_output_fn = transform_output(
        output_scaling=TorchParam(output_scale * torch.ones(1)), output_shifting=torch.zeros(1)
    )
    input_values = {fparam: torch.rand(batch_size, requires_grad=True)}
    model = get_qnn(n_qubits, depth, inputs=[fparam])
    transformed_model = get_qnn(
        n_qubits,
        depth,
        inputs=[fparam],
        input_transform=transform_input_fn,
        output_transform=transform_output_fn,
    )
    init_params = torch.rand(model.num_vparams)
    model.reset_vparams(init_params)
    transformed_model.reset_vparams(init_params)
    pred = model(input_values)
    transformed_pred = transformed_model(input_values)
    assert torch.allclose(output_scale * pred, transformed_pred)
