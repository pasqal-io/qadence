# State initialization

Several standard quantum states can be quickly initialized in `qadence`, both in statevector form as well as in block form.

## Statevector initialization

Creating uniform, all-zero or all-one:

```python exec="on" source="material-block" result="json" session="states"
from qadence import uniform_state, zero_state, one_state

n_qubits = 3
batch_size = 2

print(uniform_state(n_qubits, batch_size))
print(zero_state(n_qubits, batch_size))
print(one_state(n_qubits, batch_size))
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
