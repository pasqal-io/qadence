# QML Constructors

Besides the [arbitrary Hamiltonian constructors](hamiltonians.md), Qadence also provides a complete set of program constructors useful for digital-analog quantum machine learning programs.

## Feature-Maps

A few feature maps are directly available for feature loading,

```python exec="on" source="material-block" result="json" session="fms"
from qadence import feature_map

n_qubits = 3

fm = feature_map(n_qubits, fm_type="fourier")
print(f"{fm = }")

fm = feature_map(n_qubits, fm_type="chebyshev")
print(f"{fm = }")

fm = feature_map(n_qubits, fm_type="tower")
print(f"{fm = }")
```

## Hardware-Efficient Ansatz

### Digital HEA

Ansatze blocks for quantum machine-learning are typically built following the Hardware-Efficient Ansatz formalism (HEA). Both fully digital and digital-analog HEAs can easily be built with the `hea` function. By default, the digital version is returned:

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import hea
from qadence.draw import display

n_qubits = 3
depth = 2

ansatz = hea(n_qubits, depth)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="2,2")) # markdown-exec: hide
```

As seen above, the rotation layers are automatically parameterized, and the prefix `"theta"` can be changed with the `param_prefix` argument.

Furthermore, both the single-qubit rotations and the two-qubit entangler can be customized with the `operations` and `entangler` argument. The operations can be passed as a list of single-qubit rotations, while the entangler should be either `CNOT`, `CZ`, `CRX`, `CRY`, `CRZ` or `CPHASE`.

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import RX, RY, CPHASE

ansatz = hea(
    n_qubits = n_qubits,
    depth = depth,
    param_prefix = "phi",
    operations = [RX, RY, RX],
    entangler = CPHASE
    )
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="2,2")) # markdown-exec: hide
```

### Digital-Analog HEA

Having a truly *hardware-efficient* ansatz means that the entangling operation can be chosen according to each device's native interactions. Besides digital operations, in Qadence it is also possible to build digital-analog HEAs with the entanglement produced by the natural evolution of a set of interacting qubits, as is natural in neutral atom devices. As with other digital-analog functions, this can be controlled with the `strategy` argument which can be chosen from the [`Strategy`](../qadence/types.md) enum type. Currently, only `Strategy.DIGITAL` and `Strategy.SDAQC` are available. By default, calling `strategy = Strategy.SDAQC` will use a global entangling Hamiltonian with Ising-like NN interactions and constant interaction strength inside a `HamEvo` operation,

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import Strategy

ansatz = hea(
    n_qubits = n_qubits,
    depth = depth,
    strategy = Strategy.SDAQC
    )
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="2,2")) # markdown-exec: hide
```

Note that, by default, only the time-parameter is automatically parameterized when building a digital-analog HEA. However, as described in the [Hamiltonians tutorial](hamiltonians.md), arbitrary interaction Hamiltonians can be easily built with the `hamiltonian_factory` function, with both customized or fully parameterized interactions, and these can be directly passed as the `entangler` for a customizable digital-analog HEA.

```python exec="on" source="material-block" html="1" session="ansatz"
from qadence import hamiltonian_factory, Interaction, N, Register, hea

# Build a parameterized neutral-atom Hamiltonian following a honeycomb_lattice:
register = Register.honeycomb_lattice(1, 1)

entangler = hamiltonian_factory(
    register,
    interaction = Interaction.NN,
    detuning = N,
    interaction_strength = "e",
    detuning_strength = "n"
)

# Build a fully parameterized Digital-Analog HEA:
n_qubits = register.n_qubits
depth = 2

ansatz = hea(
    n_qubits = register.n_qubits,
    depth = depth,
    operations = [RX, RY, RX],
    entangler = entangler,
    strategy = Strategy.SDAQC
    )
from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz, size="2,2")) # markdown-exec: hide
```
