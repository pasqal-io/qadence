
There are three kinds of parameters in `qadence`:
[_**Fixed Parameter**_]: A constant with a fixed, non-trainable value (e.g. pi/2).
[_**Variational Parameter**_]: A trainable parameter which can be be optimized.
[_**Feature Parameter**_]: A non-trainable parameter which can be used to encode classical data into a quantum state.

## Parametrized Blocks
### Fixed Parameters
To pass a fixed parameter to a gate, we can simply use either python numeric types by themselves or wrapped in
a torch.tensor.
```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import RX, run

# Let's use a torch type.
block = RX(0, pi)
wf = run(block)
print(wf)

# Lets pass a simple float.
print(run(RX(0, 1.)))
```

### Variational Parameters
To parametrize a block by an angle `theta`, you can pass either a string or an instance of  `VariationalParameter` instead of a numeric type to the gate constructor:

```python exec="on" source="material-block" result="json"
from qadence import RX, run, VariationalParameter

block = RX(0, "theta")
# This is equivalent to:
block = RX(0, VariationalParameter("theta"))

wf = run(block)
print(wf)
```
In the first case in the above example, `theta` is automatically inferred as a `VariationalParameter` (i.e., a trainable one), hence we do not have to pass a value for `theta` to the `run` method since its stored within the underlying model!

### Feature Parameters

However, for `FeatureParameter`s (i.e, inputs), we always have to provide a value. And, in contrast to `VariationalParameter`s, we can also provide a batch of values.

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, run, FeatureParameter

block = RX(0, FeatureParameter("phi"))

wf = run(block, values={"phi": tensor([1., 2.])})
print(wf)
```

Now, we see that `run` returns a batch of states, one for every provided angle.
In the above case, the angle of the `RX` gate coincides with the value of the particular `FeatureParameter`.

### Multiparameter Expressions
However, an angle can itself also be a function of `Parameter`- types (fixed, trainable and non-trainable).
We can pass any sympy expression `expr: sympy.Basic` consisting of a combination of free symbols (`sympy` types) and qadence `Parameter`s to a block. This also includes, e.g., trigonometric functions!

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, Parameter, run, FeatureParameter
from sympy import sin

theta, phi = Parameter("theta"), FeatureParameter("phi")
block = RX(0, sin(theta+phi))

# Remember, to run the block, only the FeatureParameters have to be provided:
values = {"phi": tensor([1.0, 2.0])}
wf = run(block, values=values)
print(wf)
```

### Re-using Parameters

Parameters are uniquely defined by their name, so you can repeat a parameter in a composite block to
assign the same parameter to different blocks.
```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, RY, run, chain, kron

block = chain(
    kron(RX(0, "phi"), RY(1, "theta")),
    kron(RX(0, "phi"), RY(1, "theta")),
)

wf = run(block)
print(wf)
```

## Parametrized Circuits

Now, let's have a look at a variational ansatz in `qadence`.

```python exec="on" html="1"
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
block.tag = "rotations"

obs = 2*kron(*map(Z, range(3)))
block = chain(block, obs)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

## Parametrized QuantumModels

Recap:
* _**Feature**_ parameters are used for data input and encode data into a quantum state.
* _**Variational**_ parameters are trainable parameters in a variational ansatz.
* [`QuantumModel`][qadence.models.quantum_model.QuantumModel] takes an
abstract quantum circuit and makes it differentiable with respect to variational and feature
parameters.
* Both `VariationalParameter`s and `FeatureParameter`s are uniquely identified by their name.

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

print("Unique parameters in the circuit: ", circuit.unique_parameters)
```

In the circuit above, four parameters are defined but only two unique names. Therefore, there will be only one
variational parameter to be optimized.

The `QuantumModel` class also provides convenience methods to manipulate parameters.

```python exec="on" source="material-block" result="json" session="parametrized-models"
from qadence import QuantumModel, BackendName, DiffMode

model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

print(f"Number of variational parameters: {model.num_vparams}")
print(f"Current values of the variational parameters: {model.vparams}")
```

!!! note "Only provide feature parameter values to the quantum model"
    In order to `run` the variational circuit _**only feature parameter values**_ have to be provided.
	Variational parameters are stored in the model itself. If multiple feature parameters are present,
	values must be provided in batches of same length.

    ```python exec="on" source="material-block" result="json" session="parametrized-models"
    import torch

    values = {"phi": torch.rand(3)} # theta does not appear here
    wf = model.run(values)
    print(wf)
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
print(f"Unique parameters with a single HEA: {circuit.num_unique_parameters}")
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
n_params_two_heas = circuit.num_unique_parameters
print(f"Unique parameters with two stacked HEAs: {n_params_two_heas}")
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
    print(f"Unique parameters with two stacked HEAs: {n_params_two_heas}")
    ```
    ```python exec="on" html="1" session="parametrized-constructors"
    from qadence.draw import html_string # markdown-exec: hide
    print(html_string(circuit)) # markdown-exec: hide
    ```

The `hea` function will be further explored in the [QML Constructors tutorial](qml_constructors.md).

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
model = QuantumModel(circuit, observable=observable, backend="pyqtorch", diff_mode="ad")
print(model.vparams)
```

One optimization step (forward and backeward pass) can be performed and variational parameters
have been updated accordingly:

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
import torch

mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# compute forward & backward pass
optimizer.zero_grad()
loss = mse_loss(model.expectation({}), torch.zeros(1))
loss.backward()

# update the parameters
optimizer.step()
print(model.vparams)
```

## Non-unitary circuits

Qadence allows to compose with possibly non-unitary blocks.
Here is an exampl of a non-unitary block as a sum of Pauli operators with complex coefficients.

Backends which support the execution on non-unitary circuits can execute the
circuit below.

!!! warning "Currently, only the PyQTorch backend fully supports execution with non-unitary circuits."

```python exec="on" source="material-block" html="1" session="non-unitary"
from qadence import QuantumModel, QuantumCircuit, Z, X
c1 = 2.0
c2 = 2.0 + 2.0j

block = c1 * Z(0) + c2 * X(1) + c1 * c2 * (Z(2) + X(3))
circuit = QuantumCircuit(4, block)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(circuit)) # markdown-exec: hide

model = QuantumModel(circuit, backend='pyqtorch', diff_mode='ad')
print(model.run({}))
```
