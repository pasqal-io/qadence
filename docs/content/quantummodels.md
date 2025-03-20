A quantum program can be expressed and executed using the [`QuantumModel`][qadence.model.QuantumModel] type.
It serves three primary purposes:

_**Parameter handling**_: by conveniently handling and embedding the two parameter types that Qadence supports:
*feature* and *variational* (see more details in the [previous section](parameters.md)).

_**Differentiability**_: by enabling a *differentiable backend* that supports two differentiable modes: automatic differentiation (AD) and parameter shift rules (PSR). The former is used general differentiation in statevector simulators based on PyTorch and JAX. The latter is a quantum specific method used to differentiate gate parameters, and is enabled for all backends.

_**Execution**_: by defining which backend the program is expected to be executed on. Qadence supports circuit compilation to the native backend representation.

!!! note "Backends"
    The goal is for quantum models to be executed seemlessly on a number of different purpose backends: simulators, emulators or real hardware.
    By default, Qadence executes on the [PyQTorch](https://github.com/pasqal-io/PyQ) backend which implements a state vector simulator. Currently, this is the most feature rich backend. The [Pulser](https://pulser.readthedocs.io/en/stable/)
    backend is being developed, and currently supports a more limited set of functionalities (pulse sequences on programmable neutral atom arrays). The [Horqrux](https://github.com/pasqal-io/horqrux/) backend, built on JAX, is also available, but currently not supported with the `QuantumModel` interface. For more information see the [backend section](backends.md).

The base `QuantumModel` exposes the following methods:

* `QuantumModel.run()`: To extract the wavefunction after circuit execution. Not supported by all backends.
* `QuantumModel.sample()`: Sample a bitstring from the resulting quantum state after circuit execution. Supported by all backends.
* `QuantumModel.expectation()`: Compute the expectation value of an observable.

Every `QuantumModel` is an instance of a [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) that enables differentiability for its `expectation` method. For statevector simulators, AD also works for the statevector itself.

To construct a `QuantumModel`, the program block must first be initialized into a `QuantumCircuit` instance by combining it with a `Register`. An integer number can also be passed for the total number of qubits, which instantiates a `Register` automatically. The qubit register also includes topological information on the qubit layout, essential for digital-analog computations. However, we will explore that in a later tutorial. For now, let's construct a simple parametrized quantum circuit.

```python exec="on" source="material-block" result="json" session="quantum-model"
from qadence import QuantumCircuit, RX, RY, chain, kron
from qadence import FeatureParameter, VariationalParameter

theta = VariationalParameter("theta")
phi = FeatureParameter("phi")

block = chain(
    kron(RX(0, theta), RY(1, theta)),
    kron(RX(0, phi), RY(1, phi)),
)

circuit = QuantumCircuit(2, block)
unique_params = circuit.unique_parameters

print(f"{unique_params = }") # markdown-exec: hide
```

The model can then be instantiated. Similarly to the direct execution functions shown in the previous tutorial, the `run`, `sample` and `expectation` methods are available directly from the model.

```python exec="on" source="material-block" result="json" session="quantum-model"
import torch
from qadence import QuantumModel, PI, Z

observable = Z(0) + Z(1)

model = QuantumModel(circuit, observable)

values = {"phi": torch.tensor([PI, PI/2])}

wf = model.run(values)
print(f"wf = {wf.detach()}") # markdown-exec: hide
xs = model.sample(values, n_shots=100)
print(f"xs = {xs}") # markdown-exec: hide
ex = model.expectation(values)
print(f"ex = {ex.detach()}") # markdown-exec: hide
```

By default, the `forward` method of `QuantumModel` calls `model.run()`. To define custom quantum models, the best way is to inherit from `QuantumModel` and override the `forward` method, as typically done with custom PyTorch Modules.

The `QuantumModel` class provides convenience methods to manipulate parameters. Being a `torch.nn.Module`, all torch methods are also available. As shown in the example below, you can pass, check and reset the model parameters. When entering new values for the `VariationalParameter`, they must match the number of existing variables.

```python exec="on" source="material-block" result="json" session="quantum-model"
# To pass onto a torch optimizer
parameter_generator = model.parameters()

# Number of variational parameters
num_vparams = model.num_vparams

# Dictionary to see all the parameter values
params_values = model.params

# Dictionary to easily inspect variational parameters (parameters with gradient)
vparams_values = model.vparams

print(f"old {vparams_values = }") # markdown-exec: hide

# To reset current variational parameter to other values
model.reset_vparams([torch.rand(1).item()])

vparams_values = model.vparams
print(f"new {vparams_values = }") # markdown-exec: hide
```

## Backend configuration in quantum models

When initializing a quantum model, available configuration options are determined by the backend, with current support for `PyQTorch` and `Pulser`. Information on each configuration option can be found with `model.show_config` as in below example:

```python exec="on" source="material-block" result="json" session="quantum-model"
from qadence import QuantumModel, QuantumCircuit, RX, RY, kron
from qadence import FeatureParameter, VariationalParameter
from qadence import BackendConfiguration
from qadence import BackendName, DiffMode

# Create a quantum circuit
theta = VariationalParameter("theta")
phi = FeatureParameter("phi")
block = kron(RX(0, theta), RY(1, phi))
circuit = QuantumCircuit(2, block)

# Choose your backend (PYQTORCH or PULSER)
backend=BackendName.PYQTORCH
# backend=BackendName.PULSER

model = QuantumModel(circuit, backend=backend, diff_mode=DiffMode.GPSR)
# Check your available configuration options and current values
print(f"{model.show_config=}") # markdown-exec: hide
```

The configuration of the quantum model can be changed by passing `options_names` and `value` in dictionary format. You can update the existing configuration values using `model.change_config()`.

```python exec="on" source="material-block" result="json" session="quantum-model"
default_model = QuantumModel(circuit, backend=backend, diff_mode=DiffMode.GPSR)
# shows default configuration
print(f"{default_model.show_config=}") # markdown-exec: hide

# change dropout_probability from 0 to 0.1
custom_model = QuantumModel(circuit, backend=backend, diff_mode=DiffMode.GPSR, configuration = {"dropout_probability": 0.1})
# shows modified configuration
print(f"{custom_model.show_config=}") # markdown-exec: hide

custom_model.change_config({"dropout_probability": 0.3})
print(f"{custom_model.show_config=}") # markdown-exec: hide
```

## Model output

The output of a quantum model is typically encoded in the measurement of an expectation value. In Qadence, one way to customize the number of outputs is by batching the number of observables at model creation by passing a list of blocks.

```python exec="on" source="material-block" result="json" session="output"
from torch import tensor
from qadence import chain, kron, VariationalParameter, FeatureParameter
from qadence import QuantumModel, QuantumCircuit, PI, Z, RX, CNOT

theta = VariationalParameter("theta")
phi = FeatureParameter("phi")

block = chain(
    kron(RX(0, phi), RX(1, phi)),
    CNOT(0, 1)
)

circuit = QuantumCircuit(2, block)

model = QuantumModel(circuit, [Z(0), Z(0) + Z(1)])

values = {"phi": tensor(PI)}

ex = model.expectation(values)
print(f"ex = {ex.detach()}") # markdown-exec: hide
```

As mentioned in the previous tutorial, blocks can also be arbitrarily parameterized through multiplication, which allows the inclusion of trainable parameters in the definition of the observable.

```python exec="on" source="material-block" session="output"
from qadence import I, Z

a = VariationalParameter("a")
b = VariationalParameter("b")

# Magnetization with a trainable shift and scale
observable = a * I(0) + b * Z(0)

model = QuantumModel(circuit, observable)
```

## Quantum Neural Network (QNN)

The `QNN` is a subclass of the `QuantumModel` geared towards quantum machine learning and parameter optimisation. See the
[quantum machine learning section](../tutorials/qml/index.md) section or the [`QNN` API reference][qadence.ml_tools.models.QNN] for more detailed
information. There are three main differences in interface when compared with the `QuantumModel`:

- It is initialized with a list of the input parameter names, and then supports direct `torch.Tensor` inputs instead of the values dictionary shown above. The ordering of the input values should respect the order given in the input names.
- Passing an observable is mandatory.
- The `forward` method calls `model.expectation()`.

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import chain, kron, VariationalParameter, FeatureParameter
from qadence import QNN, QuantumCircuit, PI, Z, RX, RY, CNOT

theta = FeatureParameter("theta")
phi = FeatureParameter("phi")

block = chain(
    kron(RX(0, phi), RX(1, phi)),
    kron(RY(0, theta), RY(1, theta)),
    CNOT(0, 1)
)

circuit = QuantumCircuit(2, block)
observable = Z(0) + Z(1)

model = QNN(circuit, observable, inputs = ["phi", "theta"])

# "phi" = PI, PI/2, "theta" = 0.0, 1.0
values = tensor([[PI, 0.0], [PI/2, 1.0]])

ex = model(values)
print(f"ex = {ex.detach()}") # markdown-exec: hide
```
