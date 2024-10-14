Quantum programs in Qadence are constructed using a block-system, with an emphasis on composability of primitive blocks to obtain larger, composite blocks. This functional approach is different from other frameworks which follow a more object-oriented way to construct circuits and express programs.

## Primitive blocks

A [`PrimitiveBlock`][qadence.blocks.primitive.PrimitiveBlock] represents a digital or an analog time-evolution quantum operation applied to a qubit support. Programs can always be decomposed down into a sequence of `PrimitiveBlock` elements.

Two canonical examples of digital primitive blocks are the parametrized `RX` and the `CNOT` gates:

```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import chain, RX, CNOT

rx = RX(0, 0.5)
cnot = CNOT(0, 1)

block = chain(rx, cnot)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(block)) # markdown-exec: hide
```

A list of all available primitive operations can be found [here](../api/operations.md).

??? note "How to visualize blocks"

	There are two ways to display blocks in a Python interpreter: either as a tree in ASCII format using `print`:

	```python exec="on" source="material-block" result="json"
	from qadence import X, Y, kron

	kron_block = kron(X(0), Y(1))
	print(kron_block)
	```

	Or using the visualization package:

	```python exec="on" source="material-block" html="1"
	from qadence import X, Y, kron
	from qadence.draw import display

	kron_block = kron(X(0), Y(1))
	# display(kron_block)
	from qadence.draw import html_string # markdown-exec: hide
	from qadence import chain # markdown-exec: hide
	print(html_string(kron(X(0), Y(1)))) # markdown-exec: hide
	```

## Composite Blocks

Programs can be expressed by composing blocks to result in a larger [`CompositeBlock`][qadence.blocks.composite.CompositeBlock] using three fundamental operations:
_chain_, _kron_, and _add_.

- [**chain**][qadence.blocks.utils.chain] applies a set of blocks in sequence, which can have overlapping qubit supports, and results in a `ChainBlock` type. It is akin to applying a matrix product of the sub-blocks, and can also be used with the `*` operator.
- [**kron**][qadence.blocks.utils.kron] applies a set of blocks in parallel, requiring disjoint qubit support, and results in a `KronBlock` type. This is akin to applying a tensor product of the sub-blocks, and can also be used with the `@` operator.
- [**add**][qadence.blocks.utils.add] performs a direct sum of the operators, and results in an `AddBlock` type. Blocks constructed this way are typically non-unitary, as is the case for Hamiltonians which can be constructed through sums of Pauli strings. Addition can also be performed directly with the `+` operator.

```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import X, Y, chain, kron

chain_0 = chain(X(0), Y(0))
chain_1 = chain(X(1), Y(1))

kron_block = kron(chain_0, chain_1)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(kron_block)) # markdown-exec: hide
```

All composition functions support list comprehension syntax. Below we exemplify the creation of an XY Hamiltonian for qubits laid out on a line.

```python exec="on" source="material-block" result="json" session="getting_started"
from qadence import X, Y, add

def xy_int(i: int, j: int):
	return (1/2) * (X(i)@X(j) + Y(i)@Y(j))

n_qubits = 3

xy_ham = add(xy_int(i, i+1) for i in range(n_qubits-1))

print(xy_ham) # markdown-exec: hide
```

Qadence blocks can be directly translated to matrix form by calling `block.tensor()`. Note that first dimension is the batch dimension, following PyTorch conventions. This becomes relevant if the block are parameterized and batched input values are passed, as we will see later.

```python exec="on" source="material-block" result="json" session="getting_started"
from qadence import X, Y

xy = (1/2) * (X(0)@X(1) + Y(0)@Y(1))

print(xy.tensor().real)
```

For a final example of the flexibility of functional block composition, below is an implementation of the Quantum Fourier Transform on an arbitrary qubit support.

```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import H, CPHASE, PI, chain, kron

def qft_layer(qs: tuple, l: int):
	cphases = chain(CPHASE(qs[j], qs[l], PI/2**(j-l)) for j in range(l+1, len(qs)))
	return H(qs[l]) * cphases

def qft(qs: tuple):
	return chain(qft_layer(qs, l) for l in range(len(qs)))

from qadence.draw import html_string # markdown-exec: hide
print(html_string(qft((0, 1, 2)))) # markdown-exec: hide
```

Other functionalities are directly built in the block system. For example, the inverse operation can be created with the `dagger()` method.

```python exec="on" source="material-block" html="1" session="getting_started"

qft_inv = qft((0, 1, 2)).dagger()

from qadence.draw import html_string # markdown-exec: hide
print(html_string(qft_inv)) # markdown-exec: hide
```

## Digital-analog composition

In Qadence, analog operations are first-class citizens. An analog operation is one whose unitary is best described by the evolution of some hermitian generator, or Hamiltonian, acting on an arbitrary number of qubits. Qadence provides the `HamEvo` class to initialize analog operations. For a time-independent generator $\mathcal{H}$ and some time variable $t$, `HamEvo(H, t)` represents the evolution operator $\exp(-i\mathcal{H}t)$.

Analog operations constitute a generalization of digital operations, and all digital operations can also be represented as the evolution of some hermitian generator. For example, the `RX` gate is the evolution of `X`.

```python exec="on" source="material-block" session="getting_started" result="json"
from qadence import X, RX, HamEvo, PI
from torch import allclose

angle = PI/2

block_digital = RX(0, angle)

block_analog = HamEvo(0.5*X(0), angle)

print(allclose(block_digital.tensor(), block_analog.tensor()))
```

As seen in the previous section, arbitrary Hamiltonians can be constructed using Pauli operators. Their evolution can be combined with other arbitrary digital operations and incorporated into any quantum program.

```python exec="on" source="material-block" session="getting_started" html="1"
from qadence import X, Y, RX, HamEvo
from qadence import add, kron, PI

def xy_int(i: int, j: int):
	return (1/2) * (X(i)@X(j) + Y(i)@Y(j))

n_qubits = 3

xy_ham = add(xy_int(i, i+1) for i in range(n_qubits-1))

analog_evo = HamEvo(xy_ham, 1.0)

digital_block = kron(RX(i, i*PI/2) for i in range(n_qubits))

program = digital_block * analog_evo * digital_block

from qadence.draw import html_string # markdown-exec: hide
print(html_string(program)) # markdown-exec: hide
```

## Block execution

To quickly run block operations and access wavefunctions, samples or expectation values of observables, one can use the convenience functions `run`, `sample` and `expectation`.

```python exec="on" source="material-block" result="json" session="index"
from qadence import kron, add, H, Z, run, sample, expectation

n_qubits = 2

# Prepares a uniform state
h_block = kron(H(i) for i in range(n_qubits))

wf = run(h_block)
print(f"{wf = }") # markdown-exec: hide

xs = sample(h_block, n_shots=1000)
print(f"{xs = }") # markdown-exec: hide

obs = add(Z(i) for i in range(n_qubits))
ex = expectation(h_block, obs)
print(f"{ex = }") # markdown-exec: hide
```

## Execution via `QuantumCircuit` and `QuantumModel`

More fine-grained control and better performance is provided via the high-level `QuantumModel` abstraction. Quantum programs in Qadence are constructed in two steps:

1. Build a [`QuantumCircuit`][qadence.circuit.QuantumCircuit] which ties together a composite block and a register.
2. Define a [`QuantumModel`](quantummodels.md) which differentiates, compiles and executes the circuit.

Execution of more complex Qadence programs will be explored in the next tutorials.


## Adding noise to gates

It is possible to add noise to gates. Please refer to the noise tutorial [here](../tutorials/realistic_sims/noise.md).
