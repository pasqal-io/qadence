from __future__ import annotations

import itertools
from io import StringIO
from typing import Callable

import numpy as np
import torch
from matplotlib.figure import Figure

from qadence import (
    QNN,
    RX,
    RY,
    HamEvo,
    Parameter,
    QuantumCircuit,
    Z,
    add,
    chain,
    hea,
    kron,
    tag,
)
from qadence.blocks import AbstractBlock


def fig_to_html(fig: Figure) -> str:
    buffer = StringIO()
    fig.savefig(buffer, format="svg")
    return buffer.getvalue()


def hardware_efficient_ansatz(n_qubits: int = 2, depth: int = 1) -> AbstractBlock:
    return hea(n_qubits=n_qubits, depth=depth)


def digital_analog_ansatz(
    h_generator: AbstractBlock, n_qubits: int = 2, depth: int = 1, t_evo: float = 1.0
) -> AbstractBlock:
    time_evolution = HamEvo(h_generator, t_evo)

    it = itertools.count()
    ops = []
    for _ in range(depth):
        layer = kron(
            *[
                chain(*(gate(n, f"theta{next(it)}") for gate in [RX, RY, RX]))
                for n in range(n_qubits)
            ]
        )
        ops.append(chain(layer, time_evolution))
    return chain(*ops)


def qcl_circuit(n_qubits: int = 2, depth: int = 1, use_digital_analog: bool = False):
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    tag(fm, "feature_map")

    if not use_digital_analog:
        # hardware-efficient ansatz
        ansatz = hardware_efficient_ansatz(n_qubits=n_qubits, depth=depth)
    else:
        # Hamiltonian evolution ansatz (digital-analog)
        t_evo = 3.0  # length of the time evolution
        h_generator = add(
            *[Z(i) for i in range(n_qubits)]
        )  # use total magnetization as Hamiltonian
        ansatz = digital_analog_ansatz(h_generator, n_qubits=n_qubits, depth=depth, t_evo=t_evo)

    tag(ansatz, "ansatz")

    # add a final fixed layer or rotations
    fixed_layer = chain(*[RY(i, np.pi / 2) for i in range(n_qubits)])
    tag(fixed_layer, "fixed")

    blocks = [fm, ansatz, fixed_layer]
    return QuantumCircuit(n_qubits, *blocks)


def qcl_training_data(
    fn: Callable, domain: tuple = (0, 2 * np.pi), n_teacher: int = 100
) -> tuple[torch.tensor, torch.tensor]:
    start, end = domain
    x_rand_np = np.sort(np.random.uniform(low=start, high=end, size=n_teacher))
    y_rand_np = fn(x_rand_np)

    x_rand = torch.tensor(x_rand_np)
    y_rand = torch.tensor(y_rand_np)

    return x_rand, y_rand


def qcl_train_model(
    model: QNN, x_train: torch.Tensor, y_train: torch.Tensor, n_epochs: int = 50, lr: float = 1.0
) -> QNN:
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Initial loss: {mse_loss(model(x_train), y_train)}")

    for i in range(n_epochs):
        optimizer.zero_grad()

        loss = mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch {i+1} training - Loss: {loss.item()}")

    return model
