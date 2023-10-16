## Dataloaders

When using Qadence, you can supply classical data to a quantum machine learning
algorithm by using a standard PyTorch `DataLoader` instance. Qadence also provides
the `DictDataLoader` convenience class which allows
to build dictionaries of `DataLoader`s instances and easily iterate over them.

```python exec="on" source="material-block"
import torch
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import DictDataLoader

def dataloader() -> DataLoader:
    batch_size = 5
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.sin(x)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def dictdataloader() -> DictDataLoader:
    batch_size = 5

    keys = ["y1", "y2"]
    dls = {}
    for k in keys:
        x = torch.rand(batch_size, 1)
        y = torch.sin(x)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dls[k] = dataloader

    return DictDataLoader(dls)

n_epochs = 2

# iterate standard DataLoader
dl = dataloader()
for i in range(n_epochs):
    data = next(iter(dl))

# iterate DictDataLoader
ddl = dictdataloader()
for i in range(n_epochs):
    data = next(iter(ddl))

```

## Optimization routines

For training QML models, Qadence also offers a few out-of-the-box routines for optimizing differentiable
models like `QNN`s and `QuantumModel`s containing either *trainable* and/or *non-trainable* parameters
(you can refer to [the parameters tutorial](../tutorials/parameters.md) for a refresh about different parameter types):

* [`train_with_grad`][qadence.ml_tools.train_with_grad] for gradient-based optimization using PyTorch native optimizers
* [`train_gradient_free`][qadence.ml_tools.train_gradient_free] for gradient-free optimization using
the [Nevergrad](https://facebookresearch.github.io/nevergrad/) library.

These routines performs training, logging/printing loss metrics and storing intermediate checkpoints of models. In the following, we
use `train_with_grad` as example but the code can be used directly with the gradient-free routine.

As every other training routine commonly used in Machine Learning, it requires
`model`, `data` and an `optimizer` as input arguments.
However, in addition, it requires a `loss_fn` and a `TrainConfig`.
A `loss_fn` is required to be a function which expects both a model and data and returns a tuple of (loss, metrics: `<dict>`), where `metrics` is a dict of scalars which can be customized too.

```python exec="on" source="material-block"
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

The [`TrainConfig`][qadence.ml_tools.config.TrainConfig] tells `train_with_grad` what batch_size should be used,
how many epochs to train, in which intervals to print/log metrics and how often to store intermediate checkpoints.

```python exec="on" source="material-block"
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

Let's see it in action with a simple example.

### Fitting a funtion with a QNN using `ml_tools`

Let's look at a complete example of how to use `train_with_grad` now.

```python exec="on" source="material-block" html="1"
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
observable = hamiltonian_factory(n_qubits, detuning=Z)
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

n_epochs = 50

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

plt.clf() # markdown-exec: hide
plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), model(x).detach().numpy())
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

For users who want to use the low-level API of `qadence`, here is the example from above
written without `train_with_grad`.

### Fitting a function - Low-level API

```python exec="on" source="material-block"
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
observable = hamiltonian_factory(n_qubits, detuning=Z)
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
