# State initialization

Qadence offers convenience routines for preparing initial quantum states.
These routines are divided into two approaches:

- As a dense matrix.
- From a suitable quantum circuit. This is available for every backend and it should be added
in front of the desired quantum circuit to simulate.

Let's illustrate the usage of the state preparation routine.

```python exec="on" source="material-block" result="json" session="seralize"
from qadence import random_state, product_state, is_normalized, StateGeneratorType

# Random initial state.
# the default `type` is StateGeneratorType.HaarMeasureFast
state = random_state(n_qubits=2, type=StateGeneratorType.RANDOM_ROTATIONS)
print("Random initial state generated with rotations:") # markdown-exec: hide
print(f"state = {state.detach().numpy().flatten()}") # markdown-exec: hide

# Check the normalization.
assert is_normalized(state)

# Product state from a given bitstring.
# NB: Qadence follows the big endian convention.
state = product_state("01")
print("Product state corresponding to bitstring '01':") # markdown-exec: hide
print(f"state = {state.detach().numpy().flatten()}") # markdown-exec: hide
```

Now we see how to generate the product state corresponding to the one above with
a suitable quantum circuit.

```python
from qadence import product_block, tag, QuantumCircuit

state_prep_b = product_block("10")
display(state_prep_b)

# let's now prepare a circuit
state_prep_b = product_block("1000")
tag(state_prep_b, "prep")
qc_with_state_prep = QuantumCircuit(4, state_prep_b, fourier_b, hea_b)

print(html_string(qc_with_state_prep), size="4,4")) # markdown-exec: hide
```
Several standard quantum states can be conveniently initialized in Qadence, both in statevector form as well as in block form.

## Statevector initialization

Creating uniform, all-zero or all-one:

```python exec="on" source="material-block" result="json" session="states"
from qadence import uniform_state, zero_state, one_state

n_qubits = 3
batch_size = 2

niform_state = uniform_state(n_qubits, batch_size)
zero_state = zero_state(n_qubits, batch_size)
one_state = one_state(n_qubits, batch_size)
print(f"Uniform state = {uniform_state}") # markdown-exec: hide
print(f"Zero state = {zero_state}") # markdown-exec: hide
print(f"One state = {one_state}") # markdown-exec: hide
```

Creating product states:

```python exec="on" source="material-block" result="json" session="states"
from qadence import product_state, rand_product_state

# From a bitsring "100"
print(product_state("100", batch_size))

# Or a random product state
print(rand_product_state(n_qubits, batch_size))
```

Creating a GHZ state:

```python exec="on" source="material-block" result="json" session="states"
from qadence import ghz_state

print(ghz_state(n_qubits, batch_size))
```

Creating a random state uniformly sampled from a Haar measure:

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state

print(random_state(n_qubits, batch_size))
```

Custom initial states can then be passed to `run`, `sample` and `expectation` by passing the `state` argument

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state, product_state, CNOT, run

init_state = product_state("10")
final_state = run(CNOT(0, 1), state = init_state)
print(final_state)
```

## Block initialization

Not all backends support custom statevector initialization, however there are also utility functions to initialize the respective blocks:

```python exec="on" source="material-block" result="json" session="states"
from qadence import uniform_block, one_block

n_qubits = 3

print(uniform_block(n_qubits))
print(one_block(n_qubits))
```

Similarly, for product states:

```python exec="on" source="material-block" result="json" session="states"
from qadence import product_block, rand_product_block

print(product_block("100"))
print(rand_product_block(n_qubits))
```

And GHZ states:

```python exec="on" source="material-block" result="json" session="states"
from qadence import ghz_block

print(ghz_block(n_qubits))
```

Initial state blocks can simply be chained at the start of a given circuit.

## Utility functions

Some statevector utility functions are also available. We can easily create the probability mass function of a given statevector using `torch.distributions.Categorical`

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state, pmf

n_qubits = 3

state = random_state(n_qubits)
distribution = pmf(state)
print(distribution)  # markdown-exec: hide
```

We can also check if a state is normalized:

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state, is_normalized

state = random_state(n_qubits)
print(is_normalized(state))
```

Or normalize a state:

```python exec="on" source="material-block" result="json" session="states"
import torch
from qadence import normalize, is_normalized

state = torch.tensor([[1, 1, 1, 1]], dtype = torch.cdouble)
print(normalize(state))
```
