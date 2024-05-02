# Quantum machine learning constructors

Besides the [arbitrary Hamiltonian constructors](hamiltonians.md), Qadence also provides a complete set of program constructors useful for digital-analog quantum machine learning programs.

## Feature maps

The `feature_map` function can easily create several types of data-encoding blocks. The
two main types of feature maps use a Fourier basis or a Chebyshev basis.

```python exec="on" source="material-block" html="1" session="fms"
from qadence import feature_map, BasisSet, chain
from qadence.draw import display

n_qubits = 3

fourier_fm = feature_map(n_qubits, fm_type=BasisSet.FOURIER)

chebyshev_fm = feature_map(n_qubits, fm_type=BasisSet.CHEBYSHEV)

block = chain(fourier_fm, chebyshev_fm)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

A custom encoding function can also be passed with `sympy`

```python exec="on" source="material-block" html="1" session="fms"
from sympy import asin, Function

n_qubits = 3

# Using a pre-defined sympy Function
custom_fm_0 = feature_map(n_qubits, fm_type=asin)

# Creating a custom function
def custom_fn(x):
    return asin(x) + x**2

custom_fm_1 = feature_map(n_qubits, fm_type=custom_fn)

block = chain(custom_fm_0, custom_fm_1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

Furthermore, the `reupload_scaling` argument can be used to change the scaling applied to each qubit
in the support of the feature map. The default scalings can be chosen from the `ReuploadScaling` enumeration.

```python exec="on" source="material-block" html="1" session="fms"
from qadence import ReuploadScaling
from qadence.draw import display

n_qubits = 5

# Default constant value
fm_constant = feature_map(n_qubits, fm_type=BasisSet.FOURIER, reupload_scaling=ReuploadScaling.CONSTANT)

# Linearly increasing scaling
fm_tower = feature_map(n_qubits, fm_type=BasisSet.FOURIER, reupload_scaling=ReuploadScaling.TOWER)

# Exponentially increasing scaling
fm_exp = feature_map(n_qubits, fm_type=BasisSet.FOURIER, reupload_scaling=ReuploadScaling.EXP)

block = chain(fm_constant, fm_tower, fm_exp)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

A custom scaling can also be defined with a function with an `int` input and `int` or `float` output.

```python exec="on" source="material-block" html="1" session="fms"
n_qubits = 5

def custom_scaling(i: int) -> int | float:
    """Sqrt(i+1)"""
    return (i+1) ** (0.5)

# Custom scaling function
fm_custom = feature_map(n_qubits, fm_type=BasisSet.CHEBYSHEV, reupload_scaling=custom_scaling)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(fm_custom)) # markdown-exec: hide
```

To add a trainable parameter that multiplies the feature parameter inside the encoding function,
simply pass a `param_prefix` string:

```python exec="on" source="material-block" html="1" session="fms"
n_qubits = 5

fm_trainable = feature_map(
    n_qubits,
    fm_type=BasisSet.FOURIER,
    reupload_scaling=ReuploadScaling.EXP,
    param_prefix = "w",
)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(fm_trainable)) # markdown-exec: hide
```

Note that for the Fourier feature map, the encoding function is simply $f(x)=x$. For other cases, like the Chebyshev `acos()` encoding,
the trainable parameter may cause the feature value to be outside the domain of the encoding function. This will eventually be fixed
by adding range constraints to trainable parameters in Qadence.

A full description of the remaining arguments can be found in the [`feature_map` API reference][qadence.constructors.feature_map]. We provide an example below.

```python exec="on" source="material-block" html="1" session="fms"
from qadence import RY

n_qubits = 5

# Custom scaling function
fm_full = feature_map(
    n_qubits = n_qubits,
    support = tuple(reversed(range(n_qubits))), # Reverse the qubit support to run the scaling from bottom to top
    param = "x", # Change the name of the parameter
    op = RY, # Change the rotation gate between RX, RY, RZ or PHASE
    fm_type = BasisSet.CHEBYSHEV,
    reupload_scaling = ReuploadScaling.EXP,
    feature_range = (-1.0, 2.0), # Range from which the input data comes from
    target_range = (1.0, 3.0), # Range the encoder assumes as the natural range
    multiplier = 5.0, # Extra multiplier, which can also be a Parameter
    param_prefix = "w", # Add trainable parameters
)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(fm_full)) # markdown-exec: hide
```

## Hardware-efficient ansatz

Ansatze blocks for quantum machine-learning are typically built following the Hardware-Efficient Ansatz formalism (HEA).
Both fully digital and digital-analog HEAs can easily be built with the `hea` function. By default,
the digital version is returned:

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import hea
from qadence.draw import display

n_qubits = 3
depth = 2

ansatz = hea(n_qubits, depth)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz)) # markdown-exec: hide
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
print(html_string(ansatz)) # markdown-exec: hide
```

Having a truly *hardware-efficient* ansatz means that the entangling operation can be chosen according to each device's native interactions. Besides digital operations, in Qadence it is also possible to build digital-analog HEAs with the entanglement produced by the natural evolution of a set of interacting qubits, as natively implemented in neutral atom devices. As with other digital-analog functions, this can be controlled with the `strategy` argument which can be chosen from the [`Strategy`](../api/types.md) enum type. Currently, only `Strategy.DIGITAL` and `Strategy.SDAQC` are available. By default, calling `strategy = Strategy.SDAQC` will use a global entangling Hamiltonian with Ising-like $NN$ interactions and constant interaction strength,

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import Strategy

ansatz = hea(
    n_qubits,
    depth=depth,
    strategy=Strategy.SDAQC
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz)) # markdown-exec: hide
```

Note that, by default, only the time-parameter is automatically parameterized when building a digital-analog HEA. However, as described in the [Hamiltonians tutorial](hamiltonians.md), arbitrary interaction Hamiltonians can be easily built with the `hamiltonian_factory` function, with both customized or fully parameterized interactions, and these can be directly passed as the `entangler` for a customizable digital-analog HEA.

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
print(html_string(ansatz)) # markdown-exec: hide
```
## Identity-initialized ansatz

It is widely known that parametrized quantum circuits are characterized by barren plateaus, where the gradient becomes exponentially small in the number of qubits. Here we include one of many techniques that have been proposed in recent years to mitigate this effect and facilitate `QNN`s training: [Grant et al.](https://arxiv.org/abs/1903.05076) showed that initializing the weights of a `QNN` so that each block of the circuit evaluates to identity reduces the effect of barren plateaus in the initial stage of training. In a similar fashion to `hea`, such circuit can be created via calling the associated function, `identity_initialized_ansatz`:

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence.constructors import identity_initialized_ansatz
from qadence.draw import display

n_qubits = 3
depth = 2

ansatz = identity_initialized_ansatz(n_qubits, depth)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz)) # markdown-exec: hide
```
