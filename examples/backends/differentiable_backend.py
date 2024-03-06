from __future__ import annotations

import os

import numpy as np
import sympy
import torch

torch.manual_seed(42)

from qadence import (
    CNOT,
    RX,
    RY,
    Parameter,
    QuantumCircuit,
    chain,
    total_magnetization,
)
from qadence.backends.pyqtorch.backend import Backend as PyQTorchBackend
from qadence.engines.torch.differentiable_backend import DifferentiableBackend
from qadence.logger import get_script_logger

logger = get_script_logger("diff_backend")


def circuit(n_qubits):
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    fm = chain(RX(0, 3 * x), RY(1, sympy.exp(x)), RX(0, theta), RY(1, np.pi / 2))
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


if __name__ == "__main__":
    torch.manual_seed(42)
    n_qubits = 2
    batch_size = 5

    logger.info(f"Running example {os.path.basename(__file__)} with n_qubits = {n_qubits}")
    # Making circuit with AD
    circ = circuit(n_qubits)
    observable = total_magnetization(n_qubits=n_qubits)
    quantum_backend = PyQTorchBackend()
    diff_backend = DifferentiableBackend(quantum_backend, diff_mode="ad")
    diff_circ, diff_obs, embed, params = diff_backend.convert(circ, observable)

    # Running for some inputs
    values = {"x": torch.rand(batch_size, requires_grad=True)}
    wf = diff_backend.run(diff_circ, embed(params, values))
    expval = diff_backend.expectation(diff_circ, diff_obs, embed(params, values))
    dexpval_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]
    dexpval_xx = torch.autograd.grad(
        dexpval_x, values["x"], torch.ones_like(dexpval_x), create_graph=True
    )[0]
    dexpval_xxtheta = torch.autograd.grad(
        dexpval_xx,
        list(params.values())[0],
        torch.ones_like(dexpval_xx),
        retain_graph=True,
    )[0]
    dexpval_theta = torch.autograd.grad(expval, list(params.values())[0], torch.ones_like(expval))[
        0
    ]

    # Now running stuff for PSR
    diff_backend = DifferentiableBackend(quantum_backend, diff_mode="gpsr")
    expval = diff_backend.expectation(diff_circ, diff_obs, embed(params, values))
    dexpval_psr_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]
    dexpval_psr_xx = torch.autograd.grad(
        dexpval_psr_x, values["x"], torch.ones_like(dexpval_psr_x), create_graph=True
    )[0]
    dexpval_psr_xxtheta = torch.autograd.grad(
        dexpval_psr_xx,
        list(params.values())[0],
        torch.ones_like(dexpval_psr_xx),
        retain_graph=True,
    )[0]
    dexpval_psr_theta = torch.autograd.grad(
        expval, list(params.values())[0], torch.ones_like(expval)
    )[0]

    logger.info(f"Derivative with respect to 'x' with AD:  {dexpval_x}")
    logger.info(f"Derivative with respect to 'x' with PSR: {dexpval_psr_x}")
    logger.info(f"Derivative with respect to 'xx' with AD:  {dexpval_xx}")
    logger.info(f"Derivative with respect to 'xx' with PSR: {dexpval_psr_xx}")
    logger.info(f"Derivative with respect to 'xx, theta' with AD:  {dexpval_xxtheta}")
    logger.info(f"Derivative with respect to 'xx, theta' with PSR: {dexpval_psr_xxtheta}")
    logger.info(f"Derivative with respect to 'theta' with ad:  {dexpval_theta}")
    logger.info(f"Derivative with respect to 'theta' with PSR: {dexpval_psr_theta}")
