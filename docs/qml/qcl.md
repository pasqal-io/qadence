This tutorial shows how to apply `qadence` for solving a basic quantum
machine learning application: fitting a simple function with the
quantum circuit learning[^1] (QCL) algorithm.

QCL is a supervised quantum machine learning algorithm that uses a
parametrized quantum neural network to learn the behavior of an arbitrary
mathematical function using a set of function values as training data. This tutorial
shows how to fit the $\sin(x)$ function in the $[-1, 1]$ domain.

In the following, train and test data are defined.

```python exec="on" source="material-block" session="qcl" result="json"
from typing import Callable

import torch

# make sure all tensors are kept on the same device
# only available from PyTorch 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# notice that the domain does not include 1 and -1
# this avoids a singularity in the rotation angles when
# when encoding the domain points into the quantum circuit
# with a non-linear transformation (see below)
def qcl_training_data(
    domain: tuple = (-0.99, 0.99), n_points: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:

    start, end = domain

    x_rand, _ = torch.sort(torch.DoubleTensor(n_points).uniform_(start, end))
    y_rand = torch.sin(x_rand)

    return x_rand, y_rand

test_frac = 0.25
x, y = qcl_training_data()
n_test = int(len(x) * test_frac)
x_train, y_train = x[0:n_test-len(x)], y[0:n_test-len(x)]
x_test, y_test = x[n_test-len(x):], y[n_test-len(x):]
```

## Train the QCL model

Qadence provides the [`QNN`][qadence.models.qnn.QNN] convenience constructor to build a quantum neural network.
The `QNN` class needs a circuit and a list of observables; the number of feature parameters in the input circuit
determines the number of input features (i.e. the dimensionality of the classical data given as input) whereas
the number of observables determines the number of outputs of the quantum neural network.

Total qubit magnetization is used as observable:

$$
\hat{O} = \sum_i^N \hat{\sigma}_i^z
$$

In the following the observable, quantum circuit and corresponding QNN model are constructed.

```python exec="on" source="material-block" session="qcl" result="json"
import sympy
import qadence as qd
from qadence.operations import RX

n_qubits = 8

# create a simple feature map with a non-linear parameter transformation
feature_param = qd.FeatureParameter("phi")
feature_map = qd.kron(RX(i, feature_param) for i in range(n_qubits))
featre_map = qd.tag(feature_map, "feature_map")

# create a digital-analog variational ansatz using Qadence convenience constructors
ansatz = qd.hea(n_qubits, depth=n_qubits, strategy=qd.Strategy.SDAQC)
ansatz = qd.tag(ansatz, "ansatz")

# total magnetization observable
observable = qd.hamiltonian_factory(n_qubits, detuning = qd.Z)

circuit = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model = qd.QNN(circuit, [observable])
expval = model(values=torch.rand(10))
print(expval)
```

The QCL algorithm uses the output of the quantum neural network as a tunable
universal function approximator. Standard PyTorch code is used for training the QNN
using a mean-square error loss, Adam optimizer. Training is performend on the GPU
if available:

```python exec="on" source="material-block" session="qcl" result="json"

# train the model
n_epochs = 200
lr = 0.5

input_values = {"phi": x_train}
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # standard PyTorch Adam optimizer

print(f"Initial loss: {mse_loss(model(input_values), y_train)}")

y_pred_initial = model({"phi": x_test})

running_loss = 0.0
for i in range(n_epochs):

    optimizer.zero_grad()

    loss = mse_loss(model(input_values), y_train)
    loss.backward()
    optimizer.step()

    if (i+1) % 20 == 0:
        print(f"Epoch {i+1} - Loss: {loss.item()}")
```

Qadence offers some convenience functions to implement this training loop with advanced
logging and metrics track features. You can refer to [this](../qml/qml_tools.md) for more details.

The quantum model is now trained on the training data points. To determine the quality of the results,
one can check to see how well it fits the function on the test set.

```python exec="on" source="material-block" session="qcl" html="1"
import matplotlib.pyplot as plt

y_pred = model({"phi": x_test})

# convert all the results to numpy arrays for plotting
x_train_np = x_train.cpu().detach().numpy().flatten()
y_train_np = y_train.cpu().detach().numpy().flatten()
x_test_np = x_test.cpu().detach().numpy().flatten()
y_pred_initial_np = y_pred_initial.cpu().detach().numpy().flatten()
y_pred_np = y_pred.cpu().detach().numpy().flatten()

fig, _ = plt.subplots()
plt.scatter(x_train_np, y_train_np, label="Training points", marker="o", color="orange")
plt.plot(x_test_np, y_pred_initial_np, label="Initial prediction", color="green", alpha=0.5)
plt.plot(x_test_np, y_pred_np, label="Final prediction")
plt.legend()
from docs import docsutils as du # markdown-exec: hide
print(du.fig_to_html(fig)) # markdown-exec: hide
```

## References

[^1]: [Mitarai et al., Quantum Circuit Learning](https://arxiv.org/abs/1803.00745)
