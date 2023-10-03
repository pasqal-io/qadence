from __future__ import annotations

import numpy as np
import sympy
import torch

from qadence import (
    CNOT,
    RX,
    RZ,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    chain,
    total_magnetization,
)
from qadence.backend import BackendName
from qadence.backends.pytorch_wrapper import DiffMode

torch.manual_seed(42)


def circuit(n_qubits):
    x = Parameter("x", trainable=False)
    y = Parameter("y", trainable=False)

    fm = chain(RX(0, 3 * x), RZ(1, sympy.exp(y)), RX(0, np.pi / 2), RZ(1, "theta"))
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    return QuantumCircuit(n_qubits, block)


if __name__ == "__main__":
    n_qubits = 2
    batch_size = 5

    observable = total_magnetization(n_qubits)
    model = QuantumModel(
        circuit(n_qubits),
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    print(list(model.parameters()))
    nx = torch.rand(batch_size, requires_grad=True)
    ny = torch.rand(batch_size, requires_grad=True)
    values = {"x": nx, "y": ny}

    print(f"Expectation values: {model.expectation(values)}")

    # This works!
    model.zero_grad()
    loss = torch.mean(model.expectation(values))
    loss.backward()

    print("Gradients using autograd: \n")
    print("Gradient in model: \n")
    for key, param in model.named_parameters():
        print(f"{key}: {param.grad}")

    # This works too!
    print("Gradient of inputs: \n")
    print(torch.autograd.grad(torch.mean(model.expectation(values)), nx))
    print(torch.autograd.grad(torch.mean(model.expectation(values)), ny))

    # Now using PSR
    model = QuantumModel(
        circuit(n_qubits),
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.GPSR,
    )
    model.zero_grad()
    loss = torch.mean(model.expectation(values))
    loss.backward()

    print("Gradients using PSR: \n")
    print("Gradient in model: \n")
    for key, param in model.named_parameters():
        print(f"{key}: {param.grad}")

    # This works too!
    print("Gradient of inputs: \n")
    print(torch.autograd.grad(torch.mean(model.expectation(values)), nx))
    print(torch.autograd.grad(torch.mean(model.expectation(values)), ny))
