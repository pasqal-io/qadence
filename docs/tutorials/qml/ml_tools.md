## Dataloaders

When using Qadence, you can supply classical data to a quantum machine learning
algorithm by using a standard PyTorch `DataLoader` instance. Qadence also provides
the `DictDataLoader` convenience class which allows
to build dictionaries of `DataLoader`s instances and easily iterate over them.

```python exec="on" source="material-block" result="json"
import torch
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import DictDataLoader, to_dataloader


def dataloader(data_size: int = 25, batch_size: int = 5, infinite: bool = False) -> DataLoader:
    x = torch.linspace(0, 1, data_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=infinite)


def dictdataloader(data_size: int = 25, batch_size: int = 5) -> DictDataLoader:
    dls = {}
    for k in ["y1", "y2"]:
        x = torch.rand(data_size, 1)
        y = torch.sin(x)
        dls[k] = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return DictDataLoader(dls)


# iterate over standard DataLoader
for (x,y) in dataloader(data_size=6, batch_size=2):
    print(f"Standard {x = }")

# construct an infinite dataset which will keep sampling indefinitely
n_epochs = 5
dl = iter(dataloader(data_size=6, batch_size=2, infinite=True))
for _ in range(n_epochs):
    (x, y) = next(dl)
    print(f"Infinite {x = }")

# iterate over DictDataLoader
ddl = dictdataloader()
data = next(iter(ddl))
print(f"{data = }")
```

## Optimization routines

For training QML models, Qadence also offers a few out-of-the-box routines for optimizing differentiable
models, _e.g._ `QNN`s and `QuantumModel`, containing either *trainable* and/or *non-trainable* parameters
(see [the parameters tutorial](../../content/parameters.md) for detailed information about parameter types):

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

If it is desired to only the save the "best" checkpoint, the following must be ensured:

(a) `checkpoint_best_only = True` is used while creating the configuration through `TrainConfig`,
(b) `val_every` is set to a valid integer value (for example, `val_every = 10`) which controls the no. of iterations after which the validation data should be used to evaluate the model during training, which can also be set through `TrainConfig`,
(c) a validation criterion is provided through the `validation_criterion`, set through `TrainConfig` to quantify the definition of "best", and
(d) the dataloader passed to `train_grad` is of type `DictDataLoader`. In this case, it is expected that a validation dataloader is also provided along with the train dataloader since the validation data will be used to decide the "best" checkpoint. The dataloaders must be accessible with specific keys: "train" and "val".

The criterion used to decide the "best" checkpoint can be customized by `validation_criterion`, which should be a function that can take any number of arguments and return a boolean value (True or False) indicating whether some validation metric is satisfied or not. Typical choices are to return True when the validation loss (accuracy) has decreased (increased) compared to smallest (largest) value from previous iterations at which a validation check was performed.

Let's see it in action with a simple example.

### Fitting a funtion with a QNN using `ml_tools`

In Quantum Machine Learning, the general consensus is to use `complex128` precision for states and operators and `float64` precision for parameters. This is also the convention which is used in `qadence`.
However, for specific usecases, lower precision can greatly speed up training and reduce memory consumption. When using the `pyqtorch` backend, `qadence` offers the option to move a `QuantumModel` instance to a specific precision using the torch `to` syntax.

Let's look at a complete example of how to use `train_with_grad` now. Here we perform a validation check during training and use a validation criterion that checks whether the validation loss in the current iteration has decreased compared to the lowest validation loss from all previous iterations. For demonstration, the train and the validation data are kept the same here. However, it is beneficial and encouraged to keep them distinct in practice to understand model's generalization capabilities.

```python exec="on" source="material-block" html="1"
from pathlib import Path
import torch
from functools import reduce
from operator import add
from itertools import count
import matplotlib.pyplot as plt

from qadence import Parameter, QuantumCircuit, Z
from qadence import hamiltonian_factory, hea, feature_map, chain
from qadence.models import QNN
from qadence.ml_tools import  TrainConfig, train_with_grad, to_dataloader, DictDataLoader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DTYPE = torch.complex64
n_qubits = 4
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning=Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 100
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

def validation_criterion(
    current_validation_loss: float, current_best_validation_loss: float, val_epsilon: float
) -> bool:
    return current_validation_loss <= current_best_validation_loss - val_epsilon

n_epochs = 300

config = TrainConfig(
    max_iter=n_epochs,
    batch_size=batch_size,
    checkpoint_best_only=True,
    val_every=10,  # The model will be run on the validation data after every `val_every` epochs.
    validation_criterion=validation_criterion
)

fn = lambda x, degree: .05 * reduce(add, (torch.cos(i*x) + torch.sin(i*x) for i in range(degree)), 0.)
x = torch.linspace(0, 10, batch_size, dtype=torch.float32).reshape(-1, 1)
y = fn(x, 5)

data = DictDataLoader(
    {
        "train": to_dataloader(x, y, batch_size=batch_size, infinite=True),
        "val": to_dataloader(x, y, batch_size=batch_size, infinite=True),
    }
)

train_with_grad(model, data, optimizer, config, loss_fn=loss_fn,device=DEVICE, dtype=DTYPE)

plt.clf()
plt.plot(x.numpy(), y.numpy(), label='truth')
plt.plot(x.numpy(), model(x).detach().numpy(), "--", label="final", linewidth=3)
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

For users who want to use the low-level API of `qadence`, here an example written without `train_with_grad`.

### Fitting a function - Low-level API

```python exec="on" source="material-block"
from pathlib import Path
import torch
from itertools import count
from qadence.constructors import hamiltonian_factory, hea, feature_map
from qadence import chain, Parameter, QuantumCircuit, Z
from qadence.models import QNN
from qadence.ml_tools import TrainConfig

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


## Custom `train` loop

If you need custom training functionality that goes beyond what is available in
`qadence.ml_tools.train_with_grad` and `qadence.ml_tools.train_gradient_free` you can write your own
training loop based on the building blocks that are available in Qadence.

A simplified version of Qadence's train loop is defined below. Feel free to copy it and modify at
will.

```python
from typing import Callable, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, data_to_device
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.printing import print_metrics, write_tensorboard
from qadence.ml_tools.saveload import load_checkpoint, write_checkpoint


def train(
    model: Module,
    data: DataLoader,
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: str = "cpu",
    optimize_step: Callable = optimize_step,
    write_tensorboard: Callable = write_tensorboard,
) -> tuple[Module, Optimizer]:

    # Move model to device before optimizer is loaded
    model = model.to(device)

    # load available checkpoint
    init_iter = 0
    if config.folder:
        model, optimizer, init_iter = load_checkpoint(config.folder, model, optimizer)

    # initialize tensorboard
    writer = SummaryWriter(config.folder, purge_step=init_iter)

    dl_iter = iter(dataloader)

    # outer epoch loop
    for iteration in range(init_iter, init_iter + config.max_iter):
        data = data_to_device(next(dl_iter), device)
        loss, metrics = optimize_step(model, optimizer, loss_fn, data)

        if iteration % config.print_every == 0 and config.verbose:
            print_metrics(loss, metrics, iteration)

        if iteration % config.write_every == 0:
            write_tensorboard(writer, loss, metrics, iteration)

        if config.folder:
            if iteration % config.checkpoint_every == 0:
                write_checkpoint(config.folder, model, optimizer, iteration)

    # Final writing and checkpointing
    if config.folder:
        write_checkpoint(config.folder, model, optimizer, iteration)
    write_tensorboard(writer, loss, metrics, iteration)
    writer.close()

    return model, optimizer
```
