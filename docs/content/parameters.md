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

A `FeatureParameter` type can also be used. It requires an input value or a batch of values. In most cases, Qadence accepts a `values` dictionary to set the input of feature parameters.

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
from torch import tensor
from qadence import RY, PI, run, kron, FeatureParameter

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
