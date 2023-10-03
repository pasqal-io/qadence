# Constructing arbitrary Hamiltonians

A big part of working with digital-analog quantum computing is handling large analog blocks, which represent a set of interacting qubits under some interaction Hamiltonian. In `qadence` we can use the [`hamiltonian_factory`](../qadence/constructors.md) function to create arbitrary Hamiltonian blocks to be used as generators of `HamEvo` or as observables to be measured.

## Arbitrary all-to-all Hamiltonians

Arbitrary all-to-all interaction Hamiltonians can be easily created by passing the number of qubits in the first argument. The type of `interaction` can be chosen from the available ones in the [`Interaction`](../qadence/types.md) enum. Alternatively, the strings `"ZZ", "NN", "XY", "XYZ"` can also be used.

```python exec="on" source="material-block" result="json" session="hamiltonians"
from qadence import hamiltonian_factory
from qadence import N, X, Y, Z
from qadence import Interaction

n_qubits = 3

hamilt = hamiltonian_factory(n_qubits, interaction = Interaction.ZZ)

print(hamilt)
```

Single-qubit terms can also be added by passing the respective operator directly to the `detuning` argument. For example, the total magnetization is commonly used as an observable to be measured:

```python exec="on" source="material-block" result="json" session="hamiltonians"
total_mag = hamiltonian_factory(n_qubits, detuning = Z)
print(total_mag) # markdown-exec: hide
```

For further customization, arbitrary coefficients can be passed as arrays to the `interaction_strength` and `detuning_strength` arguments.

```python exec="on" source="material-block" result="json" session="hamiltonians"
n_qubits = 3

hamilt = hamiltonian_factory(
    n_qubits,
    interaction = Interaction.ZZ,
    detuning = Z,
    interaction_strength = [0.5, 0.2, 0.1],
    detuning_strength = [0.1, 0.5, -0.3]
    )
print(hamilt) # markdown-exec: hide
```

To get random interaction coefficients between -1 and 1, you can ommit `interaction_strength` and `detuning_strength` and simply pass `random_strength = True`.

Note that for passing interaction strengths as an array, you should order them in the same order obtained from the `edge` property of a Qadence [`Register`](register.md):

```python exec="on" source="material-block" result="json" session="hamiltonians"
from qadence import Register

print(Register(n_qubits).edges)
```

For one more example, let's create a transverse-field Ising model,

```python exec="on" source="material-block" session="hamiltonians"
n_qubits = 4
n_edges = int(0.5 * n_qubits * (n_qubits - 1))

z_terms = [1.0] * n_qubits
zz_terms = [2.0] * n_edges

zz_ham = hamiltonian_factory(
    n_qubits,
    interaction = Interaction.ZZ,
    detuning = Z,
    interaction_strength = zz_terms,
    detuning_strength = z_terms
    )

x_terms = [-1.0] * n_qubits
x_ham = hamiltonian_factory(n_qubits, detuning = X, detuning_strength = x_terms)

transverse_ising = zz_ham + x_ham
```


## Changing the Hamiltonian topology

We can also create arbitrary interaction topologies using the Qadence [`Register`](register.md). To do so, simply pass the register with the desired topology as the first argument.

```python exec="on" source="material-block" result="json" session="hamiltonians"
from qadence import Register

reg = Register.square(qubits_side = 2)

square_hamilt = hamiltonian_factory(reg, interaction = Interaction.NN)
print(square_hamilt) # markdown-exec: hide
```

If you wish to add specific coefficients to the Hamiltonian, you can either pass them as shown earlier, or add them to the register beforehand using the `"strength"` key.

```python exec="on" source="material-block" result="json" session="hamiltonians"

reg = Register.square(qubits_side = 2)

for i, edge in enumerate(reg.edges):
    reg.edges[edge]["strength"] = (0.5 * i) ** 2

square_hamilt = hamiltonian_factory(reg, interaction = Interaction.NN)
print(square_hamilt) # markdown-exec: hide
```

Alternatively, if your register already has saved interaction or detuning strengths but you wish to override them in the Hamiltonian creation, you can use `force_update = True`.

## Adding variational parameters

Finally, we can also easily create fully parameterized Hamiltonians by passing a string to the strength arguments. Below we create a fully parametric neutral-atom Hamiltonian,

```python exec="on" source="material-block" result="json" session="hamiltonians"
n_qubits = 3

nn_ham = hamiltonian_factory(
    n_qubits,
    interaction = Interaction.NN,
    detuning = N,
    interaction_strength = "c",
    detuning_strength = "d"
    )

print(nn_ham) # markdown-exec: hide
```
