Backends allow execution of Qadence abstract quantum circuits. They could be chosen from a variety of simulators, emulators and hardware
and can enable circuit [differentiability](https://en.wikipedia.org/wiki/Automatic_differentiation). The primary way to interact and configure
a backend is via the high-level API `QuantumModel`.

!!! note "Not all backends are equivalent"
	Not all backends support the same set of operations, especially while executing analog blocks.
	Qadence will throw descriptive errors in such cases.

## Execution backends

[_**PyQTorch**_](https://github.com/pasqal-io/PyQ): An efficient, large-scale simulator designed for
quantum machine learning, seamlessly integrated with the popular [PyTorch](https://pytorch.org/) deep learning framework for automatic differentiability.
It also offers analog computing for time-independent pulses. See [`PyQTorchBackend`][qadence.backends.pyqtorch.backend.Backend].

[_**Pulser**_](https://github.com/pasqal-io/Pulser): A Python library for pulse-level/analog control of
neutral atom devices. Execution via [QuTiP](https://qutip.org/). See [`PulserBackend`][qadence.backends.pulser.backend.Backend].

[_**Braket**_](https://github.com/aws/amazon-braket-sdk-python): A Python SDK for interacting with
quantum devices on Amazon Braket. Currently, only the devices with the digital interface of Amazon Braket
are supported and execution is performed using the local simulator. Execution on remote simulators and
quantum processing units will be available soon. See [`BraketBackend`][qadence.backends.braket.backend.Backend]

_**More**_: Proprietary Qadence extensions provide more high-performance backends based on tensor networks or differentiation engines.
For more enquiries, please contact: [`info@pasqal.com`](mailto:info@pasqal.com).

## Differentiation backend

The [`DifferentiableBackend`][qadence.backends.pytorch_wrapper.DifferentiableBackend] class enables different differentiation modes
for the given backend. This can be chosen from two types:

- Automatic differentiation (AD): available for PyTorch based backends (PyQTorch).
- Parameter Shift Rules (PSR): available for all backends. See this [section](../advanced_tutorials/differentiability.md) for more information on differentiability and PSR.

In practice, only a `diff_mode` should be provided in the `QuantumModel`. Please note that `diff_mode` defaults to `None`:

```python exec="on" source="material-block" result="json" session="diff-backend"
import sympy
import torch
from qadence import Parameter, RX, RZ, Z, CNOT, QuantumCircuit, QuantumModel, chain, BackendName, DiffMode

x = Parameter("x", trainable=False)
y = Parameter("y", trainable=False)
fm = chain(
	RX(0, 3 * x),
	RX(0, x),
	RZ(1, sympy.exp(y)),
	RX(0, 3.14),
	RZ(1, "theta")
)

ansatz = CNOT(0, 1)
block = chain(fm, ansatz)

circuit = QuantumCircuit(2, block)

observable = Z(0)

# DiffMode.GPSR is available for any backend.
# DiffMode.AD is only available for natively differentiable backends.
model = QuantumModel(circuit, observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)

# Get some values for the feature parameters.
values = {"x": (x := torch.tensor([0.5], requires_grad=True)), "y": torch.tensor([0.1])}

# Compute expectation.
exp = model.expectation(values)

# Differentiate the expectation wrt x.
dexp_dx = torch.autograd.grad(exp, x, torch.ones_like(exp))
print(f"{dexp_dx = }") # markdown-exec: hide
```

## Low-level `backend_factory` interface

Every backend in Qadence inherits from the abstract `Backend` class:
[`Backend`](../backends/backend.md) and implement the following methods:

- [`run`][qadence.backend.Backend.run]: propagate the initial state according to the quantum circuit and return the final wavefunction object.
- [`sample`][qadence.backend.Backend.sample]: sample from a circuit.
- [`expectation`][qadence.backend.Backend.expectation]: computes the expectation of a circuit given an observable.
- [`convert`][qadence.backend.Backend.convert]: convert the abstract `QuantumCircuit` object to its backend-native representation including a backend specific parameter embedding function.

Backends are purely functional objects which take as input the values for the circuit
parameters and return the desired output from a call to a method. In order to use a backend directly,
*embedded* parameters must be supplied as they are returned by the backend specific embedding function.

Here is a simple demonstration of the use of the Braket backend to execute a circuit in non-differentiable mode:

```python exec="on" source="material-block" session="low-level-braket"
from qadence import QuantumCircuit, FeatureParameter, RX, RZ, CNOT, hea, chain

# Construct a feature map.
x = FeatureParameter("x")
z = FeatureParameter("y")
fm = chain(RX(0, 3 * x), RZ(1, z), CNOT(0, 1))

# Construct a circuit with an hardware-efficient ansatz.
circuit = QuantumCircuit(3, fm, hea(3,1))
```

The abstract `QuantumCircuit` can now be converted to its native representation via the Braket
backend.

```python exec="on" source="material-block" result="json" session="low-level-braket"
from qadence import backend_factory

# Use only Braket in non-differentiable mode:
backend = backend_factory("braket")

# The `Converted` object
# (contains a `ConvertedCircuit` with the original and native representation)
conv = backend.convert(circuit)
print(f"{conv.circuit.original = }") # markdown-exec: hide
print(f"{conv.circuit.native = }") # markdown-exec: hide
```

Additionally, `Converted` contains all fixed and variational parameters, as well as an embedding
function which accepts feature parameters to construct a dictionary of *circuit native parameters*.
These are needed as each backend uses a different representation of the circuit parameters:

```python exec="on" source="material-block" result="json" session="low-level-braket"
import torch

# Contains fixed parameters and variational (from the HEA)
conv.params
print("conv.params = {") # markdown-exec: hide
for k, v in conv.params.items(): print(f"  {k}: {v}") # markdown-exec: hide
print("}") # markdown-exec: hide

inputs = {"x": torch.tensor([1., 1.]), "y":torch.tensor([2., 2.])}

# get all circuit parameters (including feature params)
embedded = conv.embedding_fn(conv.params, inputs)
print("embedded = {") # markdown-exec: hide
for k, v in embedded.items(): print(f"  {k}: {v}") # markdown-exec: hide
print("}") # markdown-exec: hide
```

Note that above the parameters keys have changed as they now address the keys on the
Braket device. A more readable embedding is provided by the PyQTorch backend:

```python exec="on" source="material-block" result="json" session="low-level-braket"
from qadence import BackendName, DiffMode
pyq_backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

# the `Converted` object
# (contains a `ConvertedCircuit` wiht the original and native representation)
pyq_conv = pyq_backend.convert(circuit)
embedded = pyq_conv.embedding_fn(pyq_conv.params, inputs)
print("embedded = {") # markdown-exec: hide
for k, v in embedded.items(): print(f"  {k}: {v}") # markdown-exec: hide
print("}") # markdown-exec: hide
```

With the embedded parameters, `QuantumModel` methods are accessible:

```python exec="on" source="material-block" result="json" session="low-level-braket"
embedded = conv.embedding_fn(conv.params, inputs)
samples = backend.run(conv.circuit, embedded)
print(f"{samples = }")
```

## Lower-level: the `Backend` representation

If there is a requirement to work with a specific backend, it is possible to access _**directly the native circuit**_.
For example, Braket noise features can be imported which are not exposed directly by Qadence.

```python exec="on" source="material-block" session="low-level-braket"
from braket.circuits import Noise

# Get the native Braket circuit with the given parameters
inputs = {"x": torch.rand(1), "y":torch.rand(1)}
embedded = conv.embedding_fn(conv.params, inputs)
native = backend.assign_parameters(conv.circuit, embedded)

# Define a noise channel
noise = Noise.Depolarizing(probability=0.1)

# Add noise to every gate in the circuit
native.apply_gate_noise(noise)
```

In order to run this noisy circuit, the density matrix simulator is needed in Braket:

```python exec="on" source="material-block" result="json" session="low-level-braket"
from braket.devices import LocalSimulator

device = LocalSimulator("braket_dm")
result = device.run(native, shots=1000).result().measurement_counts
print(result)
```
```python exec="on" source="material-block" result="json" session="low-level-braket"
print(conv.circuit.native.diagram())
```
```python exec="on" source="material-block" result="json" session="low-level-braket"
print(native.diagram())
```
