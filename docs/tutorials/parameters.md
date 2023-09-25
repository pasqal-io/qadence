
```python exec="on" html="1"
import torch
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


## Parametrized blocks

To parametrize a block simply by an angle `x` you can pass a string instead of
a fixed float to the gate constructor:

```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, run

# fixed rotation
# block = RX(0, 2.0)

# parametrised rotation
block = RX(0, "x")

wf = run(block, values={"x": torch.tensor([1.0, 2.0])})
print(wf)
```
Above you can see that `run` returns a batch of states, one for every provided angle.
You can provide any sympy expression `expr: sympy.Basic` to a block, e.g. also one with multiple
free symbols.
```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, Parameter, run

x, y = Parameter("x"), Parameter("y")
block = RX(0, x+y)

# to run the block, both parameters have to be given
values = {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([2.0, 1.0])}
wf = run(block, values=values)
print(wf)
```

Parameters are uniquely defined by their name, so you can repeat a parameter in a composite block to
assign the same parameter to different blocks.
```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, RY, run, chain, kron

block = chain(
    kron(RX(0, "phi"), RY(1, "theta")),
    kron(RX(0, "phi"), RY(1, "theta")),
)

values = {"phi": torch.rand(3), "theta": torch.tensor(3)}
wf = run(block, values=values)
print(wf)
```

## Parametrized models

In quantum models we distinguish between two kinds of parameters:

* _**Feature**_ parameters are used for data input and encode data into the quantum state.
* _**Variational**_ parameters are trainable parameters in a variational ansatz.

As a reminder, in `qadence` a [`QuantumModel`][qadence.models.quantum_model.QuantumModel] takes an
abstract quantum circuit and makes it differentiable with respect to variational and feature
parameters.

Again, both variational and feature parameters are uniquely identified by their name.
```python exec="on" source="material-block" session="parametrized-models"
from qadence import VariationalParameter, FeatureParameter, Parameter

p1 = VariationalParameter("theta")
p2 = FeatureParameter("phi")

p1_dup = VariationalParameter("theta")
p2_dup = FeatureParameter("phi")

assert p1 == p1_dup
assert p2 == p2_dup

# feature parameters are non-trainable parameters - meaning
# they can be specified via input data. The FeatureParameter
# is therefore exactly the same as a non-trainable parameter
fp = FeatureParameter("x")
assert fp == Parameter("x", trainable=False)

# variational parameters are trainable parameters
vp = VariationalParameter("y")
assert vp == Parameter("y", trainable=True)
```

Let's see them first in a quantum circuit.
```python exec="on" source="material-block" result="json" session="parametrized-models"
from qadence import QuantumCircuit, RX, RY, chain, kron

block = chain(
    kron(RX(0, p1), RY(1, p1)),
    kron(RX(0, p2), RY(1, p2)),
)

circuit = QuantumCircuit(2, block)

print("Unique parameters in the circuit: ", circuit.unique_parameters)
```

In the circuit above, we define 4 parameters but only 2 unique names. Therefore, the number of
variational parameters picked up by the optimizer in the resulting quantum model will be just 1. The
`QuantumModel` class provides some convenience methods to deal with parameters.

```python exec="on" source="material-block" result="json" session="parametrized-models"
from qadence import QuantumModel

model = QuantumModel(circuit, backend="pyqtorch", diff_mode="ad")

print(f"Number of variational parameters: {model.num_vparams}")
print(f"Current values of the variational parameters: {model.vparams}")
```

!!! note "Only provide feature parameters to the quantum model!"
    In order to `run` the variational circuit we have to _**provide only feature parameters**_, because
    the variational parameters are stored in the model itself.
    ```python exec="on" source="material-block" result="json" session="parametrized-models"
    import torch

    values = {"phi": torch.rand(3)} # theta does not appear here
    wf = model.run(values)
    print(wf)
    ```

## Usage with standard constructors

The unique parameter identification explained above is important when using built-in `qadence` block
constructors in the `qadence.constructors` such as feature maps and hardware
efficient ansatze. Let's see it in practice:

```python exec="on" source="material-block" result="json" session="parametrized-constructors"
from qadence import QuantumCircuit, hea

n_qubits = 4
depth = 2

hea1 = hea(n_qubits=n_qubits, depth=depth)
circuit = QuantumCircuit(n_qubits, hea1)
n_params_one_hea = circuit.num_unique_parameters
print(f"Unique parameters with a single HEA: {n_params_one_hea}")
```
```python exec="on" html="1" session="parametrized-constructors"
from qadence.draw import html_string
print(html_string(circuit))
```

Let's now add another HEA defined in the same way as above and create a circuit
stacking the two HEAs. As you can see below, the number of unique parameters
(and thus what gets optimized in the variational procedure) is the same since
the parameters are defined under the hood with the same names.

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

!!! warning "Avoid non-unique names!"
    The above is likely not the expected behavior when stacking two variational circuits
    together since one usually wants all the parameters to be optimized. To ensure
    this, assign a different parameter prefix for each HEA as follows.
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


## Parametric observables

In `qadence` one can define quantum observables with some (classical) optimizable parameters.  This
can be very useful for improving the convergence of some QML calculations, particularly in the
context of differentiable quantum circuits. Let's see how to define a parametrized observable:

```python exec="on" source="material-block" session="parametrized-constructors"
from qadence import VariationalParameter, Z, add, tag

s = VariationalParameter("s")
observable = add(s * Z(i) for i in range(n_qubits))
```

Create a quantum model with the parametric observable and check that the variational parameters of
the observable are among the ones of the model
```python exec="on" source="material-block" result="json" session="parametrized-constructors"
from qadence import QuantumModel, QuantumCircuit

circuit = QuantumCircuit(n_qubits, hea(n_qubits, depth))
model = QuantumModel(circuit, observable=observable, backend="pyqtorch", diff_mode="ad")
print(model.vparams)
```

We can perform one optimization step and check that the model parameters have
been updated including the observable coefficients
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

`qadence` allows to write arbitrary blocks which might also lead to non-unitary
quantum circuits. For example, let's define a non-unitary block as a sum on
Pauli operators with complex coefficients.

Backends which support the execution on non-unitary circuits can execute the
circuit below. *Currently, only PyQTorch backend fully supports execution on
non-unitary circuits.*
```python exec="on" source="material-block" html="1" session="non-unitary"
from qadence import QuantumModel, QuantumCircuit, Z, X
c1 = 2.0
c2 = 2.0 + 2.0j

block = c1 * Z(0) + c2 * X(1) + c1 * c2 * (Z(2) + X(3))
circuit = QuantumCircuit(4, block)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(circuit)) # markdown-exec: hide

model = QuantumModel(circuit)
print(model.run({}))
```
