# Constructing arbitrary Hamiltonians

At the heart of digital-analog quantum computing is the description and execution of analog blocks, which represent a set of interacting qubits under some interaction Hamiltonian.
For this purpose, Qadence relies on the [`hamiltonian_factory`](../qadence/constructors.md) function to create arbitrary Hamiltonian blocks to be used as generators of `HamEvo` or as observables to be measured.

## Arbitrary all-to-all Hamiltonians

Arbitrary all-to-all interaction Hamiltonians can be easily created by passing the number of qubits in the first argument. The type of `interaction` can be chosen from the available ones in the [`Interaction`](../qadence/types.md) enum type.

```python exec="on" source="material-block" result="json" session="hamiltonians"
from qadence import hamiltonian_factory
from qadence import N, X, Y, Z
from qadence import Interaction

n_qubits = 3

hamilt = hamiltonian_factory(n_qubits, interaction = Interaction.ZZ)

print(hamilt) # markdown-exec: hide
```

Single-qubit terms can also be added by passing the respective operator directly to the `detuning` argument. For example, the total magnetization is commonly used as an observable to be measured:

```python exec="on" source="material-block" result="json" session="hamiltonians"
total_mag = hamiltonian_factory(n_qubits, detuning = Z)
print(total_mag) # markdown-exec: hide
```

For further customization, arbitrary coefficients can be passed as arrays to the `interaction_strength` and `detuning_strength` arguments for the two-qubits and single-qubit terms respectively.

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


!!! warning "Ordering interaction strengths matters"

	When passing interaction strengths as an array, the ordering must be indentical to the one
	obtained from the `edge` property of a Qadence [`Register`](register.md):

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

!!! note "Random interaction coefficients"
	Random interaction coefficients can be chosen between -1 and 1 by simply passing `random_strength = True` instead of `detuning_strength`
	and `interaction_strength`.


## Arbitrary Hamiltonian topologies

Arbitrary interaction topologies can be created using the Qadence [`Register`](register.md).
Simply pass the register with the desired topology as the first argument to the `hamiltonian_factory`:

```python exec="on" source="material-block" result="json" session="hamiltonians"
from qadence import Register

reg = Register.square(qubits_side = 2)

square_hamilt = hamiltonian_factory(reg, interaction = Interaction.NN)
print(square_hamilt) # markdown-exec: hide
```

Custom Hamiltonian coefficients can also be added to the register beforehand using the `"strength"` key.

```python exec="on" source="material-block" result="json" session="hamiltonians"

reg = Register.square(qubits_side = 2)

for i, edge in enumerate(reg.edges):
    reg.edges[edge]["strength"] = (0.5 * i) ** 2

square_hamilt = hamiltonian_factory(reg, interaction = Interaction.NN)
print(square_hamilt) # markdown-exec: hide
```

Alternatively, if the register already stores interaction or detuning strengths, it is possible to override them in the Hamiltonian creation by using `force_update = True`.


## Adding variational parameters

Finally, fully parameterized Hamiltonians can be created by passing a string to the strength arguments:


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
