# Quantum machine learning constructors

Besides the [arbitrary Hamiltonian constructors](../tutorials/hamiltonians.md), Qadence also provides a complete set of
program constructors useful for digital-analog quantum machine learning programs.

## Feature maps

The `feature_map` function can easily create several types of data-encoding blocks. The
two main types of feature maps use a Fourier basis or a Chebyshev basis.

```python exec="on" source="material-block" html="1" session="fms"
from qadence import feature_map, BasisFeatureMap, chain
from qadence.draw import display

n_qubits = 3

fourier_fm = feature_map(n_qubits, fm_type=BasisFeatureMap.FOURIER)

chebyshev_fm = feature_map(n_qubits, fm_type=BasisFeatureMap.CHEBYSHEV)

block = chain(fourier_fm, chebyshev_fm)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block, size="6,4")) # markdown-exec: hide
```

A custom encoding function can also be passed with `sympy`

```python exec="on" source="material-block" html="1" session="fms"
from sympy import asin, Function

n_qubits = 3

# Using a pre-defined sympy Function
custom_fm_0 = feature_map(n_qubits, fm_type=asin)

# Creating a custom sub-class of Function
class custom_func(Function):
    @classmethod
    def eval(cls, x):
        return asin(x) + x**2

custom_fm_1 = feature_map(n_qubits, fm_type=custom_func)

block = chain(custom_fm_0, custom_fm_1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block, size="6,4")) # markdown-exec: hide
```

Furthermore, the `reupload_scaling` argument can be used to change the scaling applied to each qubit
in the support of the feature map. The default scalings can be chosen from the `ScalingFeatureMap` enumeration.

```python exec="on" source="material-block" html="1" session="fms"
from qadence import ScalingFeatureMap
from qadence.draw import display

n_qubits = 5

# Default constant value
fm_constant = feature_map(n_qubits, fm_type=BasisFeatureMap.FOURIER, reupload_scaling=ScalingFeatureMap.CONSTANT)

# Linearly increasing scaling
fm_tower = feature_map(n_qubits, fm_type=BasisFeatureMap.FOURIER, reupload_scaling=ScalingFeatureMap.TOWER)

# Exponentially increasing scaling
fm_exp = feature_map(n_qubits, fm_type=BasisFeatureMap.FOURIER, reupload_scaling=ScalingFeatureMap.EXP)

block = chain(fm_constant, fm_tower, fm_exp)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(block, size="6,4")) # markdown-exec: hide
```

A custom scaling can also be defined with a function with an `int` input and `int` or `float` output.

```python exec="on" source="material-block" html="1" session="fms"
n_qubits = 5

def custom_scaling(i: int) -> int | float:
    """Sqrt(i+1)"""
    return (i+1) ** (0.5)

# Custom scaling function
fm_custom = feature_map(n_qubits, fm_type=BasisFeatureMap.CHEBYSHEV, reupload_scaling=custom_scaling)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(fm_custom, size="6,4")) # markdown-exec: hide
```

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
    fm_type = BasisFeatureMap.CHEBYSHEV,
    reupload_scaling = ScalingFeatureMap.EXP,
    feature_range = (-1.0, 2.0), # Range from which the input data comes from
    target_range = (1.0, 3.0), # Range the encoder assumes as the natural range
    multiplier = 5.0 # Extra multiplier, which can also be a Parameter
)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(fm_full, size="6,4")) # markdown-exec: hide
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
print(html_string(ansatz, size="8,4")) # markdown-exec: hide
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
print(html_string(ansatz, size="8,4")) # markdown-exec: hide
```

Having a truly *hardware-efficient* ansatz means that the entangling operation can be chosen according to each device's native interactions. Besides digital operations, in Qadence it is also possible to build digital-analog HEAs with the entanglement produced by the natural evolution of a set of interacting qubits, as natively implemented in neutral atom devices. As with other digital-analog functions, this can be controlled with the `strategy` argument which can be chosen from the [`Strategy`](../qadence/types.md) enum type. Currently, only `Strategy.DIGITAL` and `Strategy.SDAQC` are available. By default, calling `strategy = Strategy.SDAQC` will use a global entangling Hamiltonian with Ising-like NN interactions and constant interaction strength,

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import Strategy

ansatz = hea(
    n_qubits,
    depth=depth,
    strategy=Strategy.SDAQC
)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="8,4")) # markdown-exec: hide
```

Note that, by default, only the time-parameter is automatically parameterized when building a digital-analog HEA. However, as described in the [Hamiltonians tutorial](../tutorials/hamiltonians.md), arbitrary interaction Hamiltonians can be easily built with the `hamiltonian_factory` function, with both customized or fully parameterized interactions, and these can be directly passed as the `entangler` for a customizable digital-analog HEA.

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
print(html_string(ansatz, size="8,4")) # markdown-exec: hide
```
