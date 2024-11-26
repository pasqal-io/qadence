Variational algorithms on noisy devices and quantum machine learning (QML)[^1] in particular are one of the main
target applications for Qadence. For this purpose, the library offers both flexible symbolic expressions for the
quantum circuit parameters via `sympy` (see [here](../../content/parameters.md) for more details) and native automatic
differentiation via integration with [PyTorch](https://pytorch.org/) deep learning framework.

Furthermore, Qadence offers a wide range of utilities for helping building and researching quantum machine learning algorithms, including:

* [a set of constructors](../../content/qml_constructors.md) for circuits commonly used in quantum machine learning such as feature maps and ansatze
* [a set of tools](ml_tools/trainer.md) for training and optimizing quantum neural networks and loading classical data into a QML algorithm

## Some simple examples

Qadence symbolic parameter interface allows to create
arbitrary feature maps to encode classical data into quantum circuits
with an arbitrary non-linear function embedding for the input values:

```python exec="on" source="material-block" result="json" session="qml"
import qadence as qd
from qadence.operations import *
import torch
from sympy import acos

n_qubits = 4

# Example feature map, also directly available with the `feature_map` function
fp = qd.FeatureParameter("phi")
fm = qd.kron(RX(i, acos(fp)) for i in range(n_qubits))

# the key in the dictionary must correspond to
# the name of the assigned to the feature parameter
inputs = {"phi": torch.rand(3)}
samples = qd.sample(fm, values=inputs)
print(f"samples = {samples[0]}")  # markdown-exec: hide
```

The [`constructors.feature_map`][qadence.constructors.feature_map] module provides
convenience functions to build commonly used feature maps where the input parameter
is encoded in the single-qubit gates rotation angle. This function will be further
demonstrated in the [QML constructors tutorial](../../content/qml_constructors.md).

Furthermore, Qadence is natively integrated with PyTorch automatic differentiation engine thus
Qadence quantum models can be used seamlessly in a PyTorch workflow.

Let's create a quantum neural network model using the feature map just defined, a
digital-analog variational ansatz ([also explained here](../../content/qml_constructors.md)) and a
simple observable $X(0) \otimes X(1)$. We use the convenience `QNN` quantum model abstraction.

```python exec="on" source="material-block" result="json" session="qml"
ansatz = qd.hea(n_qubits, strategy="sDAQC")
circuit = qd.QuantumCircuit(n_qubits, fm, ansatz)
observable = qd.kron(X(0), X(1))

model = qd.QNN(circuit, observable)

# NOTE: the `QNN` is a torch.nn.Module
assert isinstance(model, torch.nn.Module)
print(isinstance(model, torch.nn.Module)) # markdown-exec: hide
```

Differentiation works the same way as any other PyTorch module:

```python exec="on" source="material-block" result="json" session="qml"
values = {"phi": torch.rand(10, requires_grad=True)}

# the forward pass of the quantum model returns the expectation
# value of the input observable
out = model(values)
print(f"Quantum model output: \n{out}\n")  # markdown-exec: hide

# you can compute the gradient with respect to inputs using
# PyTorch autograd differentiation engine
dout = torch.autograd.grad(out, values["phi"], torch.ones_like(out), create_graph=True)[0]
print(f"First-order derivative w.r.t. the feature parameter: \n{dout}")

# you can also call directly a backward pass to compute derivatives with respect
# to the variational parameters and use it for implementing variational
# optimization
out.sum().backward()
```

To run QML on real devices, Qadence offers generalized parameter shift rules (GPSR) [^2]
for arbitrary quantum operations which can be selected when constructing the
`QNN` model:

```python exec="on" source="material-block" result="json" session="qml"
model = qd.QNN(circuit, observable, diff_mode="gpsr")
out = model(values)

dout = torch.autograd.grad(out, values["phi"], torch.ones_like(out), create_graph=True)[0]
print(f"First-order derivative w.r.t. the feature parameter: \n{dout}")
```

See [here](../advanced_tutorials/differentiability.md) for more details on how the parameter
shift rules implementation works in Qadence.

## References

[^1] Schuld, Petruccione, Machine learning on Quantum Computers, Springer Nature (2021)

[^2]: [Kyriienko et al., General quantum circuit differentiation rules](https://arxiv.org/abs/2108.01218)
