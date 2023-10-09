Qadence base `Parameter` type is a subtype of `sympy.Symbol`. There are three kinds of parameter subtypes used:

- _**Fixed Parameter**_: A constant with a fixed, non-trainable value (_e.g._ $\dfrac{\pi}{2}$).
- _**Variational Parameter**_: A trainable parameter which can be be optimized.
- _**Feature Parameter**_: A non-trainable parameter which can be used to encode classical data into a quantum state.

## Fixed Parameters

To pass a fixed parameter to a gate (or any parametrizable block), one can simply use either Python numeric types or wrapped in
a `torch.Tensor`.

```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import RX, run

# Let's use a torch type.
block = RX(0, pi)
wf = run(block)
print(f"{wf = }") # markdown-exec: hide

# Let's pass a simple float.
block = RX(0, 1.)
wf = run(block)
print(f"{wf = }") # markdown-exec: hide
```

## Variational Parameters

To parametrize a block by an angle `theta`, either a Python `string` or an instance of  `VariationalParameter` can be passed instead of a numeric type to the gate constructor:

```python exec="on" source="material-block" result="json"
from qadence import RX, run, VariationalParameter

block = RX(0, "theta")
# This is equivalent to:
block = RX(0, VariationalParameter("theta"))

wf = run(block)
print(f"{wf = }") # markdown-exec: hide
```

In the first case in the above example, `theta` is automatically inferred as a `VariationalParameter` (_i.e._ trainable). It is initialized to a random value for the purposes of execution. In the context of a `QuantumModel`, there is no need to pass a value for `theta` to the `run` method since it is stored within the underlying model parameter dictionary.

## Feature Parameters

`FeatureParameter` types (_i.e._ inputs), always need to be provided with a value or a batch of values as a dictionary:

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, run, FeatureParameter

block = RX(0, FeatureParameter("phi"))

wf = run(block, values={"phi": tensor([1., 2.])})
print(f"{wf = }") # markdown-exec: hide
```

Now, `run` returns a batch of states, one for every provided angle which coincides with the value of the particular `FeatureParameter`.

## Multiparameter Expressions

However, an angle can itself be an expression `Parameter` types of any kind.
As such, any sympy expression `expr: sympy.Basic` consisting of a combination of free symbols (_i.e._ `sympy` types) and Qadence `Parameter` can
be passed to a block, including trigonometric functions.

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, Parameter, run, FeatureParameter
from sympy import sin

theta, phi = Parameter("theta"), FeatureParameter("phi")
block = RX(0, sin(theta+phi))

# Remember, to run the block, only FeatureParameter values have to be provided:
values = {"phi": tensor([1.0, 2.0])}
wf = run(block, values=values)
print(f"{wf = }") # markdown-exec: hide
```

## Parameters Redundancy

Parameters are uniquely defined by their name and redundancy is allowed in composite blocks to
assign the same value to different blocks.

```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, RY, run, chain, kron

block = chain(
    kron(RX(0, "phi"), RY(1, "theta")),
    kron(RX(0, "phi"), RY(1, "theta")),
)

wf = run(block)  # Same random initialization for all instances of phi and theta.
print(f"{wf = }") # markdown-exec: hide
```

## Parametrized Circuits

Now, let's have a look at the construction of a variational ansatz which composes `FeatureParameter` and `VariationalParameter` types:

```python exec="on" source="material-block" html="1"
import sympy
from qadence import RX, RY, RZ, CNOT, Z, run, chain, kron, FeatureParameter, VariationalParameter

phi = FeatureParameter("phi")
theta = VariationalParameter("theta")

block = chain(
    kron(
        RX(0, phi/theta),
        RY(1, theta*2),
        RZ(2, sympy.cos(phi)),
    ),
    kron(
        RX(0, phi),
        RY(1, theta),
        RZ(2, phi),
    ),
    kron(
        RX(0, phi),
        RY(1, theta),
        RZ(2, phi),
    ),
    kron(
        RX(0, phi + theta),
        RY(1, theta**2),
        RZ(2, sympy.cos(phi)),
    ),
    chain(CNOT(0,1), CNOT(1,2))
)
block.tag = "Rotations"

obs = 2*kron(*map(Z, range(3)))
block = chain(block, obs)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(block, size="4,4")) # markdown-exec: hide
```

Please note the different colors for the parametrization with different types. The default palette assigns light blue for `VariationalParameter`, light green for `FeatureParameter` and shaded red for observables.

## Parametrized QuantumModels

As a quick reminder: `FeatureParameter` are used for data input and data encoding into a quantum state.
`VariationalParameter` are trainable parameters in a variational ansatz. When used within a [`QuantumModel`][qadence.models.quantum_model.QuantumModel], an abstract quantum circuit is made differentiable with respect to both variational and feature
parameters which are uniquely identified by their name.

```python exec="on" source="material-block" session="parametrized-models"
from qadence import FeatureParameter, Parameter, VariationalParameter

# Feature parameters are non-trainable parameters.
# Their primary use is input data encoding.
fp = FeatureParameter("x")
assert fp == Parameter("x", trainable=False)

# Variational parameters are trainable parameters.
# Their primary use is for optimization.
vp = VariationalParameter("y")
assert vp == Parameter("y", trainable=True)
```

Let's construct a parametric quantum circuit.

```python exec="on" source="material-block" result="json" session="parametrized-models"
from qadence import QuantumCircuit, RX, RY, chain, kron

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

In the circuit above, four parameters are defined but only two unique names. Therefore, there will be only one
variational parameter to be optimized.

The `QuantumModel` class also provides convenience methods to manipulate parameters.

```python exec="on" source="material-block" result="json" session="parametrized-models"
from qadence import QuantumModel, BackendName, DiffMode

model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
num_vparams = model.num_vparams
vparams_values = model.vparams

print(f"{num_vparams = }") # markdown-exec: hide
print(f"{vparams_values = }") # markdown-exec: hide
```

!!! note "Only provide feature parameter values to the quantum model"
    In order to `run` the variational circuit _**only feature parameter values**_ have to be provided.
	Variational parameters are stored in the model itself. If multiple feature parameters are present,
	values must be provided in batches of same length.

    ```python exec="on" source="material-block" result="json" session="parametrized-models"
    import torch

    values = {"phi": torch.rand(3)} # theta does not appear here
    wf = model.run(values)
    print(f"{wf = }") # markdown-exec: hide
    ```

## Standard constructors

The unique parameter identification is relevant when using built-in Qadence block
constructors in the `qadence.constructors` module such as feature maps and hardware
efficient ansatze (HEA).

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
from qadence import QuantumCircuit, hea

n_qubits = 4
depth = 2

hea1 = hea(n_qubits=n_qubits, depth=depth)
circuit = QuantumCircuit(n_qubits, hea1)
num_unique_parameters = circuit.num_unique_parameters
print(f"Unique parameters with a single HEA: {num_unique_parameters}") # markdown-exec: hide
```
```python exec="on" html="1" session="parametrized-constructors"
from qadence.draw import html_string
print(html_string(circuit))
```

A new circuit can be created by adding another identical HEA. As expected, the number of unique parameters
is the same.

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
hea2 = hea(n_qubits=n_qubits, depth=depth)

circuit = QuantumCircuit(n_qubits, hea1, hea2)
num_unique_params_two_heas = circuit.num_unique_parameters
print(f"Unique parameters with two stacked HEAs: {num_unique_params_two_heas}") # markdown-exec: hide
```
```python exec="on" html="1" session="parametrized-constructors"
from qadence.draw import html_string # markdown-exec: hide
print(html_string(circuit)) # markdown-exec: hide
```

!!! warning "Avoid non-unique names by prefixing"
    A parameter prefix for each HEA can be passed as follows:

    ```python exec="on" source="material-block" result="json" session="parametrized-constructors"
    hea1 = hea(n_qubits=n_qubits, depth=depth, param_prefix="p1")
    hea2 = hea(n_qubits=n_qubits, depth=depth, param_prefix="p2")

    circuit = QuantumCircuit(n_qubits, hea1, hea2)
    n_params_two_heas = circuit.num_unique_parameters
    print(f"Unique parameters with two stacked HEAs: {n_params_two_heas}") # markdown-exec: hide
    ```
    ```python exec="on" html="1" session="parametrized-constructors"
    from qadence.draw import html_string # markdown-exec: hide
    print(html_string(circuit)) # markdown-exec: hide
    ```

The `hea` function will be further explored in the [QML Constructors tutorial](tutorials.qml_tools).

## Parametric observables

In Qadence, one can define quantum observables with classical optimizable parameters to
improve the convergence of QML calculations. This is particularly useful for differentiable quantum circuits.

```python exec="on" source="material-block" session="parametrized-constructors"
from qadence import VariationalParameter, Z, add, tag

s = VariationalParameter("s")
observable = add(s * Z(i) for i in range(n_qubits))
```

Now, a quantum model can be created with the parametric observable.
The observable variational parameters are included among the model ones.

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
from qadence import QuantumModel, QuantumCircuit

circuit = QuantumCircuit(n_qubits, hea(n_qubits, depth))
model = QuantumModel(circuit, observable=observable)
print(f"Variational parameters = {model.vparams}") # markdown-exec: hide
```

One optimization step (forward and backward pass) can be performed using built-in `torch` functionalities. Variational parameters
can be checked to have been updated accordingly:

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
import torch

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Compute forward & backward pass
optimizer.zero_grad()
loss = mse_loss(model.expectation({}), torch.zeros(1))
loss.backward()

# Update the parameters and check the parameters.
optimizer.step()
print(f"Variational parameters = {model.vparams}") # markdown-exec: hide
```

## Non-unitary circuits

Qadence allows to compose with non-unitary blocks.
Here is an example of a non-unitary block as a sum of Pauli operators with complex coefficients.

!!! warning "Currently, only the `PyQTorch` backend fully supports execution with non-unitary circuits."

```python exec="on" source="material-block" result="json" session="non-unitary"
from qadence import QuantumModel, QuantumCircuit, Z, X
c1 = 2.0
c2 = 2.0 + 2.0j

block = c1 * Z(0) + c2 * X(1) + c1 * c2 * (Z(2) + X(3))
circuit = QuantumCircuit(4, block)

model = QuantumModel(circuit)  # BackendName.PYQTORCH and DiffMode.AD by default.
print(f"wf = {model.run({})}") # markdown-exec: hide
```
