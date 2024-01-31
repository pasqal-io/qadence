from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

import qadence as qd
from qadence.qinfo_tools import QuantumNaturalGradient

# make sure all tensors are kept on the same device
# only available from PyTorch 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

n_qubits = 4


def qcl_training_data(
    domain: tuple = (0, 2 * torch.pi), n_points: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    start, end = domain

    x_rand, _ = torch.sort(torch.DoubleTensor(n_points).uniform_(start, end))
    y_rand = torch.sin(x_rand)

    return x_rand, y_rand


x, y = qcl_training_data()

# random train/test split of the dataset
train_subset, test_subset = random_split(x, [0.75, 0.25])
train_ind = sorted(train_subset.indices)
test_ind = sorted(test_subset.indices)

x_train, y_train = x[train_ind], y[train_ind]
x_test, y_test = x[test_ind], y[test_ind]


# create a simple feature map to encode the input data
feature_param = qd.FeatureParameter("phi")
feature_map = qd.kron(qd.RX(i, feature_param) for i in range(n_qubits))
feature_map = qd.tag(feature_map, "feature_map")

# create a digital-analog variational ansatz using Qadence convenience constructors
ansatz = qd.hea(n_qubits, depth=n_qubits)
ansatz = qd.tag(ansatz, "ansatz")

# Observable
observable = qd.hamiltonian_factory(n_qubits, detuning=qd.Z)

# Create separate circuits for the two optimizers
circuit_adam = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
circuit_qng = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
circuit_qng_spsa = qd.QuantumCircuit(n_qubits, feature_map, ansatz)

model_adam = qd.QNN(circuit_adam, [observable])
model_qng = qd.QNN(circuit_qng, [observable])
model_qng_spsa = qd.QNN(circuit_qng_spsa, [observable])

circ_params_qng = [param for param in model_qng.parameters() if param.requires_grad]
circ_params_qng_spsa = [param for param in model_qng_spsa.parameters() if param.requires_grad]

# Train with ADAM
n_epochs_adam = 40
lr_adam = 0.01
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = torch.optim.Adam(model_adam.parameters(), lr=lr_adam)  # standard PyTorch Adam optimizer
loss_adam = []
print(f"Initial loss: {mse_loss(model_adam(values=x_train), y_train)}")
for i in range(n_epochs_adam):
    optimizer.zero_grad()
    loss = mse_loss(model_adam(values=x_train).squeeze(), y_train)
    loss_adam.append(float(loss))
    loss.backward()
    optimizer.step()
    if (i + 1) % 2 == 0:
        print(f"Epoch {i+1} - Loss: {loss.item()}")

# Train with QNG
n_epochs_qng = 20
lr_qng = 0.01
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = QuantumNaturalGradient(circ_params_qng, lr=lr_qng, circuit=circuit_qng)
loss_qng = []
print(f"Initial loss: {mse_loss(model_qng(values=x_train), y_train)}")
for i in range(n_epochs_qng):
    optimizer.zero_grad()
    loss = mse_loss(model_qng(values=x_train).squeeze(), y_train)
    loss_qng.append(float(loss))
    loss.backward()
    optimizer.step()
    if (i + 1) % 2 == 0:
        print(f"Epoch {i+1} - Loss: {loss.item()}")


# Train with QNG-SPSA
n_epochs_qng_spsa = 20
lr_qng_spsa = 5e-4
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = QuantumNaturalGradient(
    circ_params_qng_spsa, lr=lr_qng_spsa, circuit=circuit_qng_spsa, approximation="spsa"
)
loss_qng_spsa = []
print(f"Initial loss: {mse_loss(model_qng_spsa(values=x_train), y_train)}")
for i in range(n_epochs_qng_spsa):
    optimizer.zero_grad()
    loss = mse_loss(model_qng_spsa(values=x_train).squeeze(), y_train)
    loss_qng_spsa.append(float(loss))
    loss.backward()
    optimizer.step()
    if (i + 1) % 2 == 0:
        print(f"Epoch {i+1} - Loss: {loss.item()}")


# Plot losses
fig, _ = plt.subplots()
plt.plot(range(n_epochs_adam), loss_adam, label="Adam optimizer")
plt.plot(range(n_epochs_qng), loss_qng, label="QNG optimizer")
plt.plot(range(n_epochs_qng_spsa), loss_qng_spsa, label="QNG-SPSA optimizer")
plt.legend()
plt.xlabel("Training epochs")
plt.ylabel("Loss")
plt.show()
