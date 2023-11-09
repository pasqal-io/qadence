from __future__ import annotations

import sys
from timeit import timeit

import matplotlib.pyplot as plt
import torch

from qadence import (
    AnalogRX,
    AnalogRZ,
    DiffMode,
    FeatureParameter,
    QuantumCircuit,
    QuantumModel,
    Register,
    VariationalParameter,
    Z,
    add,
    chain,
    expectation,
    wait,
)

pi = torch.pi
SHOW_PLOTS = sys.argv[1] == "show" if len(sys.argv) == 2 else False


def plot(x, y, **kwargs):
    xnp = x.detach().cpu().numpy().flatten()
    ynp = y.detach().cpu().numpy().flatten()
    return plt.plot(xnp, ynp, **kwargs)


def scatter(x, y, **kwargs):
    xnp = x.detach().cpu().numpy().flatten()
    ynp = y.detach().cpu().numpy().flatten()
    return plt.scatter(xnp, ynp, **kwargs)


# two qubit register
reg = Register.from_coordinates([(0, 0), (0, 12)])

# analog ansatz with input parameter
t = FeatureParameter("t")

block = chain(
    AnalogRX(pi / 2),
    AnalogRZ(t),
    # NOTE: for a better fit, manually set delta
    # AnalogRot(duration=1000 / (6 * torch.pi) * t, delta=6 * torch.pi),  # RZ
    wait(1000 * VariationalParameter("theta", value=0.5)),
    AnalogRX(pi / 2),
)

# observable
obs = add(Z(i) for i in range(reg.n_qubits))


# define problem
x_train = torch.linspace(0, 6, steps=30)
y_train = -0.64 * torch.sin(x_train + 0.33) + 0.1

y_pred_initial = expectation(reg, block, obs, values={"t": x_train})


# define quantum model; including digital-analog emulation
circ = QuantumCircuit(reg, block)
model = QuantumModel(circ, obs, diff_mode=DiffMode.GPSR)

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)


def loss_fn(x_train, y_train):
    return mse_loss(model.expectation({"t": x_train}).squeeze(), y_train)


print(loss_fn(x_train, y_train))
print(timeit(lambda: loss_fn(x_train, y_train), number=5))

# train
n_epochs = 200

for i in range(n_epochs):
    optimizer.zero_grad()

    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f"Epoch {i+1:0>3} - Loss: {loss.item()}")

# visualize
y_pred = model.expectation({"t": x_train})

plt.figure()
scatter(x_train, y_train, label="Training points", marker="o", color="green")
plot(x_train, y_pred_initial, label="Initial prediction")
plot(x_train, y_pred, label="Final prediction")


plt.legend()
if SHOW_PLOTS:
    plt.show()

assert loss_fn(x_train, y_train) < 0.05
