Qadence provides a flexible parameter system built on top of Sympy. Parameters can be of different types:

- Fixed parameter: a constant with a fixed, non-trainable value (_e.g._ $\dfrac{\pi}{2}$).
- Variational parameter: a trainable parameter which will be automatically picked up by the optimizer.
- Feature parameter: a non-trainable parameter which can be used to pass input values.

## Fixed parameters

Passing fixed parameters to blocks can be done by simply passing a Python numeric type or a `torch.Tensor`.

```python exec="on" source="material-block" result="json"
import torch
from qadence import RX, run, PI

wf = run(RX(0, torch.tensor(PI)))
print(f"{wf = }") # markdown-exec: hide

wf = run(RX(0, PI))
print(f"{wf = }") # markdown-exec: hide
```

## Variational parameters

To parametrize a block a `VariationalParameter` instance is required. In most cases Qadence also accepts a Python string, which will be used to automatically initialize a `VariationalParameter`:

```python exec="on" source="material-block" result="json"
from qadence import RX, run, VariationalParameter

block = RX(0, VariationalParameter("theta"))
block = RX(0, "theta")  # Equivalent

wf = run(block)
print(f"{wf = }") # markdown-exec: hide
```

By calling `run`, a random value for `"theta"` is initialized at execution. In a `QuantumModel`, variational parameters are stored in the underlying model parameter dictionary.

## Feature parameters

A `FeatureParameter` type can also be used, which will require an input value or a batch of values. In most cases, Qadence accepts a `values` dictionary to set the input of feature parameters.

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, PI, run, FeatureParameter

block = RX(0, FeatureParameter("phi"))

wf = run(block, values = {"phi": tensor([PI, PI/2])})
print(f"{wf = }") # markdown-exec: hide
```

Since a batch of input values was passed, the `run` function returns a batch of output states. Note that `FeatureParameter("x")` and `VariationalParameter("x")` are simply aliases for `Parameter("x", trainable = False)` and `Parameter("x", trainable = True)`.

## Multiparameter expressions and analog integration

The integration with Sympy becomes useful when one wishes to write arbitrary parameter compositions. Parameters can also be used as scaling coefficients in the block system, which is essential when defining arbitrary analog operations.

```python exec="on" source="material-block" result="json"
from torch import tensor
from qadence import RX, Z, HamEvo, PI
from qadence import VariationalParameter, FeatureParameter, run
from sympy import sin

theta, phi = VariationalParameter("theta"), FeatureParameter("phi")

# Arbitrary parameter composition
expr = PI * sin(theta + phi)

# Use as unitary gate arguments
gate = RX(0, expr)

# Or as scaling coefficients for Hermitian operators
h_op = expr * (Z(0) @ Z(1))

wf = run(gate * HamEvo(h_op, phi), values = {"phi": tensor(PI)})
print(f"{wf = }") # markdown-exec: hide
```

## Parameter redundancy

Parameters are uniquely defined by their name and redundancy is allowed in composite blocks to assign the same value to different blocks. This is useful, for example, when defining layers of rotation gates typically used as feature maps.

```python exec="on" source="material-block" result="json"
import torch
from qadence import RY, run, kron, FeatureParameter

n_qubits = 3

param = FeatureParameter("phi")

block = kron(RY(i, (i+1) * param) for i in range(n_qubits))

wf = run(block, values = {"phi": tensor(PI)})
print(f"{wf = }") # markdown-exec: hide
```

## Parametrized circuits

Let's look at a final example of an arbitrary composition of digital and analog parameterized blocks:

```python exec="on" source="material-block" html="1"
import sympy
from qadence import RX, RY, RZ, CNOT, CPHASE, Z, HamEvo
from qadence import run, chain, add, kron, FeatureParameter, VariationalParameter, PI

n_qubits = 3

phi = FeatureParameter("Φ")
theta = VariationalParameter("θ")

rotation_block = kron(
    RX(0, phi/theta),
    RY(1, theta*2),
    RZ(2, sympy.cos(phi))
)
digital_entangler = CNOT(0, 1) * CPHASE(1, 2, PI)

hamiltonian = add(theta * (Z(i) @ Z(i+1)) for i in range(n_qubits-1))

analog_evo = HamEvo(hamiltonian, phi)

program = chain(rotation_block, digital_entangler, analog_evo)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(program)) # markdown-exec: hide
```

Please note the different colors for the parametrization with different types. The default palette assigns blue for `VariationalParameter`, green for `FeatureParameter`, orange for numeric values, and shaded red for non-parametric gates.

## Parametrized QuantumModels

When used within a [`QuantumModel`][qadence.models.quantum_model.QuantumModel], an abstract quantum circuit is made differentiable with respect to both variational and feature parameters which are uniquely identified by their name.

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
num_vparams = model.num_vparams # get the number of variational parameters
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

The `hea` function will be further explored in the [QML Constructors tutorial](../tutorials/qml/qml_constructors.md).

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

Qadence allows composing with non-unitary blocks.
Here is an example of a non-unitary block as a sum of Pauli operators with complex coefficients.

!!! warning "Currently, only the `PyQTorch` backend fully supports execution of non-unitary circuits."

```python exec="on" source="material-block" result="json" session="non-unitary"
from qadence import QuantumModel, QuantumCircuit, Z, X
c1 = 2.0
c2 = 2.0 + 2.0j

block = c1 * Z(0) + c2 * X(1) + c1 * c2 * (Z(2) + X(3))
circuit = QuantumCircuit(4, block)

model = QuantumModel(circuit)  # BackendName.PYQTORCH and DiffMode.AD by default.
print(f"wf = {model.run({})}") # markdown-exec: hide
```
