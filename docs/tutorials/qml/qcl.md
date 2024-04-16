This tutorial shows how to apply `qadence` for solving a basic quantum
machine learning application: fitting a simple function with the
quantum circuit learning[^1] (QCL) algorithm.

QCL is a supervised quantum machine learning algorithm that uses a
parametrized quantum neural network to learn the behavior of an arbitrary
mathematical function using a set of function values as training data. This tutorial
shows how to fit the $\sin(x)$ function in the $[-1, 1]$ domain.

In the following, train and test data are defined.

```python exec="on" source="material-block" session="qcl"
import torch
from torch.utils.data import random_split

# make sure all tensors are kept on the same device
# only available from PyTorch 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def qcl_training_data(
    domain: tuple = (0, 2*torch.pi), n_points: int = 200
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
import qadence as qd

n_qubits = 4

# create a simple feature map to encode the input data
feature_param = qd.FeatureParameter("phi")
feature_map = qd.kron(qd.RX(i, feature_param) for i in range(n_qubits))
feature_map = qd.tag(feature_map, "feature_map")

# create a digital-analog variational ansatz using Qadence convenience constructors
ansatz = qd.hea(n_qubits, depth=n_qubits)
ansatz = qd.tag(ansatz, "ansatz")

# total qubit magnetization observable
observable = qd.hamiltonian_factory(n_qubits, detuning=qd.Z)

circuit = qd.QuantumCircuit(n_qubits, feature_map, ansatz)
model = qd.QNN(circuit, [observable])
expval = model(values=torch.rand(10))
print(expval) # markdown-exec: hide
```

The QCL algorithm uses the output of the quantum neural network as a tunable
universal function approximator. Standard PyTorch code is used for training the QNN
using a mean-square error loss, Adam optimizer. Training is performend on the GPU
if available:

```python exec="on" source="material-block" session="qcl" result="json"
n_epochs = 100
lr = 0.25

input_values = {"phi": x_train}
mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # standard PyTorch Adam optimizer

print(f"Initial loss: {mse_loss(model(values=x_train), y_train)}")
y_pred_initial = model(values=x_test)

for i in range(n_epochs):

    optimizer.zero_grad()

    # given a `n_batch` number of input points and a `n_observables`
    # number of input observables to measure, the QNN returns
    # an output of the following shape: [n_batch x n_observables]
    # given that there is only one observable, a squeeze is applied to get
    # a 1-dimensional tensor
    loss = mse_loss(model(values=x_train).squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (i+1) % 20 == 0:
        print(f"Epoch {i+1} - Loss: {loss.item()}")

assert loss.item() < 1e-3
```

Qadence offers some convenience functions to implement this training loop with advanced
logging and metrics track features. You can refer to [this tutorial](../qml/ml_tools.md) for more details.

The quantum model is now trained on the training data points. To determine the quality of the results,
one can check to see how well it fits the function on the test set.

```python exec="on" source="material-block" session="qcl" html="1"
import matplotlib.pyplot as plt

y_pred = model({"phi": x_test})

# convert all the results to numpy arrays for plotting
x_train_np = x_train.cpu().detach().numpy().flatten()
y_train_np = y_train.cpu().detach().numpy().flatten()
x_test_np = x_test.cpu().detach().numpy().flatten()
y_test_np = y_test.cpu().detach().numpy().flatten()
y_pred_initial_np = y_pred_initial.cpu().detach().numpy().flatten()
y_pred_np = y_pred.cpu().detach().numpy().flatten()

fig, _ = plt.subplots()
plt.scatter(x_test_np, y_test_np, label="Test points", marker="o", color="orange")
plt.plot(x_test_np, y_pred_initial_np, label="Initial prediction", color="green", alpha=0.5)
plt.plot(x_test_np, y_pred_np, label="Final prediction")
plt.legend()
from docs import docsutils as du # markdown-exec: hide
print(du.fig_to_html(fig)) # markdown-exec: hide
```

## References

[^1]: [Mitarai et al., Quantum Circuit Learning](https://arxiv.org/abs/1803.00745)
