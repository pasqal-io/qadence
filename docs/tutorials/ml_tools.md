`qadence` also offers a out-of-the-box training routine called `train_with_grad`
for optimizing fully-differentiable models like `QNN`s and `QuantumModel`s containing either *trainable* and/or *non-trainable* parameters (i.e., inputs). Feel free to [refresh your memory about different parameter types](/tutorials/parameters).

## ML tools Basics

`train_with_grad` performs training, logging/printing loss metrics and storing intermediate checkpoints of models.

As every other training routine commonly used in Machine Learning, it requires
`model`, `data` and an `optimizer` as input arguments.
However, in addition, it requires a `loss_fn` and a `TrainConfig`.
A `loss_fn` is required to be a function which expects both a model and data and returns a tuple of (loss, metrics: dict), where `metrics` is a dict of scalars which can be customized too.

```python exec="on" source="material-block" result="json"
import torch
from itertools import count
cnt = count()
criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data[0], data[1]
    out = model(x)
    loss = criterion(out, y)
    return loss, {}

```

The `TrainConfig` [qadence.ml_tools.config] tells `train_with_grad` what batch_size should be used, how many epochs to train, in which intervals to print/log metrics and how often to store intermediate checkpoints.

```python exec="on" source="material-block" result="json"
from qadence.ml_tools import TrainConfig

batch_size = 5
n_epochs = 100

config = TrainConfig(
    folder="some_path/",
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
)

```
## Fitting a funtion with a QNN using ml_tools

Let's look at a complete example of how to use `train_with_grad` now.

```python exec="on" source="material-block" result="json"
from pathlib import Path
import torch
from itertools import count
from qadence.constructors import hamiltonian_factory, hea, feature_map
from qadence import chain, Parameter, QuantumCircuit, Z
from qadence.models import QNN
from qadence.ml_tools import train_with_grad, TrainConfig
import matplotlib.pyplot as plt

n_qubits = 2
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning = Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 1
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

cnt = count()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data[0], data[1]
    out = model(x)
    loss = criterion(out, y)
    return loss, {}

tmp_path = Path("/tmp")

n_epochs = 5

config = TrainConfig(
    folder=tmp_path,
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
)

batch_size = 25

x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
y = torch.sin(x)

train_with_grad(model, (x, y), optimizer, config, loss_fn=loss_fn)

plt.plot(y.numpy())
plt.plot(model(input_values).detach().numpy())

```

For users who want to use the low-level API of `qadence`, here is the example from above
written without `train_with_grad`.

## Fitting a function - Low-level API

```python exec="on" source="material-block" result="json"
from pathlib import Path
import torch
from itertools import count
from qadence.constructors import hamiltonian_factory, hea, feature_map
from qadence import chain, Parameter, QuantumCircuit, Z
from qadence.models import QNN
from qadence.ml_tools import train_with_grad, TrainConfig

n_qubits = 2
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning = Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 1
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epochs=50
cnt = count()

tmp_path = Path("/tmp")

config = TrainConfig(
    folder=tmp_path,
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
)

x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
y = torch.sin(x)

for i in range(n_epochs):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

```
