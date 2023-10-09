# Quantum Machine Learning Constructors

Besides the [arbitrary Hamiltonian constructors](hamiltonians.md), Qadence also provides a complete set of program constructors useful for digital-analog quantum machine learning programs.

## Feature Maps

A few feature maps are directly available for feature loading,

```python exec="on" source="material-block" result="json" session="fms"
from qadence import feature_map

n_qubits = 3

fm = feature_map(n_qubits, fm_type="fourier")
print(f"Fourier = {fm}") # markdown-exec: hide

fm = feature_map(n_qubits, fm_type="chebyshev")
print(f"Chebyshev {fm}") # markdown-exec: hide

fm = feature_map(n_qubits, fm_type="tower")
print(f"Tower {fm}") # markdown-exec: hide
```

## Hardware-Efficient Ansatz

Ansatze blocks for quantum machine-learning are typically built following the Hardware-Efficient Ansatz formalism (HEA). Both fully digital and digital-analog HEAs can easily be built with the `hea` function. By default, the digital version is returned:

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import hea
from qadence.draw import display

n_qubits = 3
depth = 2

ansatz = hea(n_qubits, depth)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="4,4")) # markdown-exec: hide
```

As seen above, the rotation layers are automatically parameterized, and the prefix `"theta"` can be changed with the `param_prefix` argument.

Furthermore, both the single-qubit rotations and the two-qubit entangler can be customized with the `operations` and `entangler` argument. The operations can be passed as a list of single-qubit rotations, while the entangler should be either `CNOT`, `CZ`, `CRX`, `CRY`, `CRZ` or `CPHASE`.

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import RX, RY, CPHASE

ansatz = hea(
    n_qubits=n_qubits,
    depth=depth,
    param_prefix="phi",
    operations=[RX, RY, RX],
    entangler=CPHASE
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="4,4")) # markdown-exec: hide
```

Having a truly *hardware-efficient* ansatz means that the entangling operation can be chosen according to each device's native interactions. Besides digital operations, in Qadence it is also possible to build digital-analog HEAs with the entanglement produced by the natural evolution of a set of interacting qubits, as is natural in neutral atom devices. As with other digital-analog functions, this can be controlled with the `strategy` argument which can be chosen from the [`Strategy`](qadence.types.Strategy) enum type. Currently, only `Strategy.DIGITAL` and `Strategy.SDAQC` are available. By default, calling `strategy = Strategy.SDAQC` will use a global entangling Hamiltonian with Ising-like NN interactions and constant interaction strength inside a `HamEvo` operation,

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import Strategy

ansatz = hea(
    n_qubits,
    depth=depth,
    strategy=Strategy.SDAQC
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="4,4")) # markdown-exec: hide
```

Note that, by default, only the time-parameter is automatically parameterized when building a digital-analog HEA. However, as described in the [Hamiltonians tutorial](tutorials.hamiltonians), arbitrary interaction Hamiltonians can be easily built with the `hamiltonian_factory` function, with both customized or fully parameterized interactions, and these can be directly passed as the `entangler` for a customizable digital-analog HEA.

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import hamiltonian_factory, Interaction, N, Register, hea

# Build a parameterized neutral-atom Hamiltonian following a honeycomb_lattice:
register = Register.honeycomb_lattice(1, 1)

entangler = hamiltonian_factory(
    register,
    interaction=Interaction.NN,
    detuning=N,
    interaction_strength="e",
    detuning_strength="n"
)

# Build a fully parameterized Digital-Analog HEA:
n_qubits = register.n_qubits
depth = 2

ansatz = hea(
    n_qubits=register.n_qubits,
    depth=depth,
    operations=[RX, RY, RX],
    entangler=entangler,
    strategy=Strategy.SDAQC
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="4,4")) # markdown-exec: hide
```
Qadence also offers a out-of-the-box training routine called `train_with_grad`
for optimizing fully-differentiable models like `QNN`s and `QuantumModel`s containing either *trainable* and/or *non-trainable* parameters (i.e., inputs). Feel free to [refresh your memory about different parameter types](/tutorials/parameters).

## Machine Learning Tools

`train_with_grad` performs training, logging/printing loss metrics and storing intermediate checkpoints of models.

As every other training routine commonly used in Machine Learning, it requires
`model`, `data` and an `optimizer` as input arguments.
However, in addition, it requires a `loss_fn` and a `TrainConfig`.
A `loss_fn` is required to be a function which expects both a model and data and returns a tuple of (loss, metrics: `<dict>`), where `metrics` is a dict of scalars which can be customized too.

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
## Fitting a funtion with a QNN using `ml_tools`

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
