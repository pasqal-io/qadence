Backends in Qadence are what make an abstract quantum circuit executable on different kinds of
emulators and hardware **and** they make our circuits
[differentiable](https://en.wikipedia.org/wiki/Automatic_differentiation).  Under the hood they are
what the `QuantumModel`s use.

In order to use the different backends you do not have to know anything about their implementation
details. Qadence conveniently lets you specify the backend you want to run on in the `QuantumModel`.
Some backends do not support all operations, for example the Braket backend cannot execute analog
blocks, but Qadence will throw descriptive errors when you try to execute unsupported blocks.

## Execution backends

[_**PyQTorch**_](https://github.com/pasqal-io/PyQ): An efficient, large-scale emulator designed for
quantum machine learning, seamlessly integrated with the popular PyTorch deep learning framework for automatic differentiability.
Implementation details: [`PyQTorchBackend`][qadence.backends.pyqtorch.backend.Backend].

[_**Pulser**_](https://pulser.readthedocs.io/en/stable/): Library for pulse-level/analog control of
neutral atom devices. Emulator via QuTiP.

[_**Braket**_](https://github.com/aws/amazon-braket-sdk-python):  A Python SDK for interacting with
quantum devices on Amazon Braket. Currently, only the devices with the digital interface of Amazon Braket
are supported and execution is performed using the local simulator. Execution on remote simulators and
quantum processing units will be available soon.

_**More**_: In the premium version of Qadence we provide even more backends such as a tensor network
emulator. For more info write us at: [`info@pasqal.com`](mailto:info@pasqal.com).

## Differentiation backends

[`DifferentiableBackend`][qadence.backends.pytorch_wrapper.DifferentiableBackend] is the class
that takes care of applying the different differentiation modes.
In your scripts you only have to provide a `diff_mode` in the `QuantumModel` via

You can make any circuit differentiable using efficient and general parameter shift rules (PSRs).
See [link](...) for more information on differentiability and PSR.
```python
QuantumModel(..., diff_mode="gpsr")
```


??? note "Set up a circuit with feature parameters (defines the `circuit` function used below)."
    ```python exec="on" source="material-block" session="diff-backend"
    import sympy
    from qadence import Parameter, RX, RZ, CNOT, QuantumCircuit, chain

    def circuit(n_qubits: int):
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
        return QuantumCircuit(2, block)
    ```

!!! note "Make any circuit differentiable via PSR diff mode."
    ```python exec="on" source="material-block" result="json" session="diff-backend"
    import torch
    from qadence import QuantumModel, Z

    circuit = circuit(n_qubits=2)
    observable = Z(0)

    # you can freely choose any backend with diff_mode="psr"
    # diff_mode="ad" will only work with natively differentiable backends.
    model = QuantumModel(circuit, observable, backend="pyqtorch", diff_mode="gpsr")

    # get some values for the feature parameters
    values = {"x": (x := torch.tensor([0.5], requires_grad=True)), "y": torch.tensor([0.1])}

    # compute expectation
    e = model.expectation(values)

    # differentiate it!
    g = torch.autograd.grad(e, x, torch.ones_like(e))
    print(f"{g = }") # markdown-exec: hide
    ```


## Low-level `Backend` Interface

Every backend in `qadence` inherits from the abstract `Backend` class:
[`Backend`](../backends/backend.md).

All backends implement these methods:

- [`run`][qadence.backend.Backend.run]: Propagate the initial state according to the quantum circuit and return the final wavefunction object.
- [`sample`][qadence.backend.Backend.sample]: Sample from a circuit.
- [`expectation`][qadence.backend.Backend.expectation]: Computes the expectation of a circuit given
    an observable.
- [`convert`][qadence.backend.Backend.convert]: Convert the abstract `QuantumCircuit` object to
    its backend-native representation including a backend specific parameter embedding function.

The quantum backends are purely functional objects which take as input the values of the circuit
parameters and return the desired output.  In order to use a backend directly, you need to supply
*embedded* parameters as they are returned by the backend specific embedding function.

To demonstrate how to use a backend directly we will construct a simple `QuantumCircuit` and run it
on the Braket backend.

```python exec="on" source="material-block" session="low-level-braket"
from qadence import QuantumCircuit, FeatureParameter, RX, RZ, CNOT, hea, chain

# construct a featuremap
x = FeatureParameter("x")
z = FeatureParameter("y")
fm = chain(RX(0, 3 * x), RZ(1, z), CNOT(0, 1))

# circuit with hardware-efficient ansatz
circuit = QuantumCircuit(3, fm, hea(3,1))
```

The abstract `QuantumCircuit` can now be converted to its native representation via the Braket
backend.

```python exec="on" source="material-block" result="json" session="low-level-braket"
from qadence import backend_factory

# use only Braket without differentiable backend by supplying `diff_mode=None`:
backend = backend_factory("braket", diff_mode=None)

# the `Converted` object
# (contains a `ConvertedCircuit` wiht the original and native representation)
conv = backend.convert(circuit)
print(f"{conv.circuit.original = }")
print(f"{conv.circuit.native = }")
```

Additionally `Converted` contains all fixed and variational parameters, as well as an embedding
function which accepts feature parameters to construct a dictionary of *circuit native parameters*. These are needed since each backend uses a different representation of the circuit parameters under the hood:

```python exec="on" source="material-block" result="json" session="low-level-braket"
import torch

# contains fixed parameters and variational (from the HEA)
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

Note that above the keys of the parameters have changed, because they now address the keys on the
Braket device. A more readable embedding is the embedding of the PyQTorch backend:
```python exec="on" source="material-block" result="json" session="low-level-braket"
pyq_backend = backend_factory("pyqtorch", diff_mode="ad")

# the `Converted` object
# (contains a `ConvertedCircuit` wiht the original and native representation)
pyq_conv = pyq_backend.convert(circuit)
embedded = pyq_conv.embedding_fn(pyq_conv.params, inputs)
print("embedded = {") # markdown-exec: hide
for k, v in embedded.items(): print(f"  {k}: {v}") # markdown-exec: hide
print("}") # markdown-exec: hide
```

With the embedded parameters we can call the methods we know from the `QuantumModel` like
`backend.run`:
```python exec="on" source="material-block" result="json" session="low-level-braket"
embedded = conv.embedding_fn(conv.params, inputs)
samples = backend.run(conv.circuit, embedded)
print(f"{samples = }")
```

### Even lower-level: Use the backend representation directly

If you have to do things that are not currently supported by `qadence` but only by a specific backend
itself, you can always _**work directly with the native circuit**_.
For example, we can couple `qadence` directly with Braket noise features which are not exposed directly by Qadence.
```python exec="on" source="material-block" session="low-level-braket"
from braket.circuits import Noise

# get the native Braket circuit with the given parameters
inputs = {"x": torch.rand(1), "y":torch.rand(1)}
embedded = conv.embedding_fn(conv.params, inputs)
native = backend.assign_parameters(conv.circuit, embedded)

# define a noise channel
noise = Noise.Depolarizing(probability=0.1)

# add noise to every gate in the circuit
native.apply_gate_noise(noise)
```

The density matrix simulator is needed in Braket to run this noisy circuit. Let's do the rest of the
example using Braket directly.
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
