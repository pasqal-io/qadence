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
print("Random initial state generated with rotations:\n") # markdown-exec: hide
print(f"state = {state.detach().numpy().flatten()}\n") # markdown-exec: hide

# Check the normalization.
assert is_normalized(state)

# Product state from a given bitstring.
# NB: Qadence follows the big endian convention.
state = product_state("01")
print("Product state corresponding to bitstring '01':\n") # markdown-exec: hide
print(f"state = {state.detach().numpy().flatten()}") # markdown-exec: hide
```

Now we see how to generate the product state corresponding to the one above with
a suitable quantum circuit.

```python  exec="on" source="material-block" html="1"
from qadence import product_block, tag, hea, QuantumCircuit
from qadence.draw import display

state_prep_block = product_block("01")
display(state_prep_block)

# Let's now prepare a circuit.
n_qubits = 4

state_prep_block = product_block("0001")
tag(state_prep_block, "Prep block")

circuit_block = tag(hea(n_qubits, depth = 2), "Circuit block")

qc_with_state_prep = QuantumCircuit(n_qubits, state_prep_block, circuit_block)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(qc_with_state_prep), size="4,4") # markdown-exec: hide
```
Several standard quantum states can be conveniently initialized in Qadence, both in statevector form as well as in block form as shown in following.

## State vector initialization

Qadence offers a number of constructor functions for state vector preparation.

```python exec="on" source="material-block" result="json" session="states"
from qadence import uniform_state, zero_state, one_state

n_qubits = 3
batch_size = 2

uniform_state = uniform_state(n_qubits, batch_size)
zero_state = zero_state(n_qubits, batch_size)
one_state = one_state(n_qubits, batch_size)
print("Uniform state = \n") # markdown-exec: hide
print(f"{uniform_state}") # markdown-exec: hide
print("Zero state = \n") # markdown-exec: hide
print(f"{zero_state}") # markdown-exec: hide
print("One state = \n") # markdown-exec: hide
print(f"{one_state}") # markdown-exec: hide
```

As already seen, product states can be easily created, even in batches:

```python exec="on" source="material-block" result="json" session="states"
from qadence import product_state, rand_product_state

# From a bitsring "100"
prod_state = product_state("100", batch_size)
print("Product state = \n") # markdown-exec: hide
print(f"{prod_state}\n") # markdown-exec: hide

# Or a random product state
rand_state = rand_product_state(n_qubits, batch_size)
print("Random state = \n") # markdown-exec: hide
print(f"{rand_state}") # markdown-exec: hide
```

Creating a GHZ state:

```python exec="on" source="material-block" result="json" session="states"
from qadence import ghz_state

ghz = ghz_state(n_qubits, batch_size)

print("GHZ state = \n") # markdown-exec: hide
print(f"{ghz}") # markdown-exec: hide
```

Creating a random state uniformly sampled from a Haar measure:

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state

rand_haar_state = random_state(n_qubits, batch_size)

print("Random state from Haar = \n") # markdown-exec: hide
print(f"{rand_haar_state}") # markdown-exec: hide
```

Custom initial states can then be passed to either `run`, `sample` and `expectation` through the `state` argument

```python exec="on" source="material-block" result="json" session="states"
from qadence import random_state, product_state, CNOT, run

init_state = product_state("10")
final_state = run(CNOT(0, 1), state=init_state)

print(f"Final state = {final_state}") # markdown-exec: hide
```

## Block initialization

Not all backends support custom statevector initialization, however previous utility functions have their counterparts to initialize the respective blocks:

```python exec="on" source="material-block" result="json" session="states"
from qadence import uniform_block, one_block

n_qubits = 3

uniform_block = uniform_block(n_qubits)
print(uniform_block) # markdown-exec: hide

one_block = one_block(n_qubits)
print(one_block) # markdown-exec: hide
```

Similarly, for product states:

```python exec="on" source="material-block" result="json" session="states"
from qadence import product_block, rand_product_block

product_block = product_block("100")
print(product_block) # markdown-exec: hide

rand_product_block = rand_product_block(n_qubits)
print(rand_product_block) # markdown-exec: hide
```

And GHZ states:

```python exec="on" source="material-block" result="json" session="states"
from qadence import ghz_block

ghz_block = ghz_block(n_qubits)
print(ghz_block) # markdown-exec: hide
```

Initial state blocks can simply be chained at the start of a given circuit.

## Utility functions

Some state vector utility functions are also available. We can easily create the probability mass function of a given statevector using `torch.distributions.Categorical`

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
