#!/bin/python
from __future__ import annotations

import numpy as np
import sympy
import torch
torch.set_default_device("cuda")
torch.manual_seed(42)
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()
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
from qadence.types import BackendName, DiffMode
from qadence.logger import get_script_logger

logger = get_script_logger("diff_backend")

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
    logger.info(f"Running quantum models example with n_qubits {2}")
    observable = total_magnetization(n_qubits)
    model = QuantumModel(
        circuit(n_qubits),
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    model.to("cuda")
    logger.info(list(model.parameters()))
    nx = torch.rand(batch_size, requires_grad=True)
    ny = torch.rand(batch_size, requires_grad=True)
    values = {"x": nx, "y": ny}

    logger.info(f"Expectation values: {model.expectation(values)}")

    # This works!
    model.zero_grad()
    loss = torch.mean(model.expectation(values))
    loss.backward()

    logger.info("Gradients using autograd: \n")
    logger.info("Gradient in model: \n")
    for key, param in model.named_parameters():
        logger.info(f"{key}: {param.grad}")

    # This works too!
    logger.info("Gradient of inputs: \n")
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), nx))
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), ny))

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

    logger.info("Gradients using PSR: \n")
    logger.info("Gradient in model: \n")
    for key, param in model.named_parameters():
        logger.info(f"{key}: {param.grad}")

    # This works too!
    logger.info("Gradient of inputs: \n")
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), nx))
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), ny))

    # Finally, lets try ADJOINT
    model = QuantumModel(
        circuit(n_qubits),
        observable=observable,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.ADJOINT,
    )
    model.zero_grad()
    loss = torch.mean(model.expectation(values))
    loss.backward()

    logger.info("Gradients using ADJOINT: \n")
    logger.info("Gradient in model: \n")
    for key, param in model.named_parameters():
        logger.info(f"{key}: {param.grad}")

    logger.info("Gradient of inputs: \n")
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), nx))
    logger.info(torch.autograd.grad(torch.mean(model.expectation(values)), ny))
