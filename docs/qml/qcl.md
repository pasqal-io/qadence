In this tutorial, we show how to apply `qadence` for solving a basic quantum
machine learning application: fitting a simple function with the
quantum circuit learning (QCL) algorithm.

Quantum circuit learning [^1] is a supervised quantum machine learning algorithm that uses
parametrized quantum neural networks to learn the behavior of an arbitrary
mathematical function starting from some training data extracted from it. We
choose the function

For this tutorial, we show how to fit the $sin(x)$ function in the domain $[-1, 1]$.

Let's start with defining training and test data.

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
The `QNN` class needs a circuit and a list of observables; both the number of feature parameters and the number
of observables in the list must be equal to the number of desired outputs of the quantum neural network.

As observable, we use the total qubit magnetization leveraging a convenience constructor provided by `qadence`:

$$
\hat{O} = \sum_i^N \hat{\sigma}_i^z
$$

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
observable = qd.total_magnetization(n_qubits)

circuit = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model = qd.QNN(circuit, [observable])
expval = model(values=torch.rand(10))
print(expval)
```

The QCL algorithm uses the output of the quantum neural network as a tunable
function approximator. We can use standard PyTorch code for training the QNN
using a mean-square error loss, the Adam optimizer and also train on the GPU
if any is available:

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

The quantum model is now trained on the training data points. Let's see how well it fits the
function on the test set.

```python exec="on" source="material-block" session="qcl" result="json"
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

## Find the extremum of the QCL model

After training the QCL model, we can find the minimum or maximum (extremum) of
the fitted function by frozing the optimal model parameters and train on the
input parameters instead. We call this procedure extremization.

The extremization procedure aims at finding the value of the feature parameter `phi` which
corresponds to the maximum (or minimum) of the fitted function.

In order to do so we need to:

* retrieve the optimal parameters of the trained model
* freeze the variational ansatz and make the feature map trainable such that
it can be optimized to find the maximum of the fitted function. For this purpose, we
use the convenience `set_trainable` routine which takes an input sub-block and makes
all its parameters either trainable or not
* train a new extremization model to find the maximum

```python exec="on" source="material-block" result="json" session="qcl" result="json"
import numpy as np
from qadence.transpile import set_trainable

# get the optimal model parameters
# these will become the input parameters
# for the extremization procedure
optimal_parameters = model.vparams

# freeze ansatz and make feature map trainable
# the lookup is done using the given tags
set_trainable(circuit.get_blocks_by_tag("feature_map"), value=True)
set_trainable(circuit.get_blocks_by_tag("ansatz"), value=False)

# make another QNN for extremization
extremize_model = qd.QNN(circuit, observable)

# perform extremization
lr_extr = 1.0
n_epochs_extr = 100
optimizer = torch.optim.Adam(extremize_model.parameters(), lr=lr_extr)

running_loss = 0.0
for i in range(n_epochs_extr):
    optimizer.zero_grad()

    # find the maximum by simply taking the output of the model
    # as the loss function
    loss = -1.0 * extremize_model(optimal_parameters)
    loss.backward()
    optimizer.step()
    if (i + 1) % 20 == 0:
        print(f"Epoch {i+1} extremization - Loss: {loss.item()}")

x_max = extremize_model.vparams["phi"]
y_max = model(x_max)
x_max_np = x_max.detach().numpy().flatten()
y_max_np = y_max.detach().numpy().flatten()

plt.figure()
plt.scatter(x_max_np, y_max_np, label="Extrema", marker="*", color="orange", sizes=[100])
plt.plot(x_test_np, y_pred_np, label="Final prediction")
plt.plot(x_test_np, np.sin(x_test_np), label="Analytical solution")
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


## References

[^1]: [Mitarai et al., Quantum Circuit Learning](https://arxiv.org/abs/1803.00745)
