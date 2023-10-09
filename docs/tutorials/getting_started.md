Quantum programs in Qadence are constructed via a block-system, with an emphasis on composability of
*primitive* blocks to obtain larger, *composite* blocks. This functional approach is different from other frameworks
which follow a more object-oriented way to construct circuits and express programs.

??? note "How to visualize blocks"

	There are two ways to display blocks in a Python interpreter: either as a tree in ASCII format using `print`:

	```python exec="on" source="material-block" result="json"
	from qadence import X, Y, kron

	kron_block = kron(X(0), Y(1))
	print(kron_block)
	```

	Or using the visualization package which opens an interactive window:

	```python exec="on" source="material-block" html="1"
	from qadence import X, Y, kron
	#from visualization import display

	kron_block = kron(X(0), Y(1))
	#display(kron_block)

	from qadence.draw import html_string # markdown-exec: hide
	from qadence import chain # markdown-exec: hide
	print(html_string(kron(X(0), Y(1)))) # markdown-exec: hide
	```

## Primitive blocks

A [`PrimitiveBlock`][qadence.blocks.primitive.PrimitiveBlock] represents a digital or an analog time-evolution quantum operation applied to a qubit support.
Programs can always be decomposed down into a sequence of `PrimitiveBlock` elements.

Two canonical examples of digital primitive blocks are the parametrized `RX` and the `CNOT` gates:

```python exec="on" source="material-block" html="1"
from qadence import RX

# A rotation gate on qubit 0 with a fixed numerical parameter.
rx_gate = RX(0, 0.5)

from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(rx_gate))) # markdown-exec: hide
```

```python exec="on" source="material-block" html="1"
from qadence import CNOT

# A CNOT gate with control on qubit 0 and target on qubit 1.
cnot_gate = CNOT(0, 1)
from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(cnot_gate))) # markdown-exec: hide
```

A list of all instances of primitive blocks (also referred to as *operations*) can be found [here](../qadence/operations.md).

## Composite Blocks

Programs can be expressed by composing blocks to result in a larger [`CompositeBlock`][qadence.blocks.composite.CompositeBlock] using three fundamental operations:
_chain_, _kron_, and _add_.

- [**chain**][qadence.blocks.utils.chain] applies a set of blocks in sequence on the *same or overlapping qubit supports* and results in a `ChainBlock` type.
It is akin to applying a matrix product of the sub-blocks with the `*` operator.

```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, chain

# Chaining on the same qubit using a call to the function.
chain_x = chain(X(0), X(0))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(chain_x)) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1" session="i-xx"
# Chaining on different qubits using the operator overload.
# Identical to the kron operation.
chain_xx = X(0) * X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(chain_xx)) # markdown-exec: hide
```

- [**kron**][qadence.blocks.utils.kron] applies a set of blocks in parallel (simultaneously) on *disjoint qubit support* and results in a `KronBlock` type. This is akin to applying a tensor product of the sub-blocks with the `@` operator.

```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, kron

kron_xx = kron(X(0), X(1))  # Equivalent to X(0) @ X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(kron_xx)) # markdown-exec: hide
```

For the digital case, it should be noted that `kron` and `chain` are semantically equivalent up to the diagrammatic representation as `chain` implicitly fills blank wires with identities.
However, Qadence also supports *analog* blocks, for which composing sequentially or in parallel becomes non-equivalent. More
about analog blocks can be found in the [digital-analog](/digital_analog_qc/analog-basics) section.

- [**add**][qadence.blocks.utils.add] sums the corresponding matrix of
each sub-block and results in a `AddBlock` type which can be used to construct Pauli operators.
Please note that `AddBlock` can give rise to non-unitary computations that might not be supported by all backends.

??? note "Get the matrix of a block"
    It is always possible to retrieve the matrix representation of a block by calling the `block.tensor()` method.
	Please note that the returned tensor contains a batch dimension for the purposes of block parametrization.

    ```python exec="on" source="material-block" result="json" session="i-xx"
    print(f"X(0) * X(0) tensor = {chain_x.tensor()}") # markdown-exec: hide
    print(f"X(0) @ X(1) tensor = {chain_xx.tensor()}") # markdown-exec: hide
    ```

```python exec="on" source="material-block" result="json"
from qadence import X, Z

xz = X(0) + Z(0)
print(xz.tensor())
```

Finally, it is possible to tag blocks with human-readable names:

```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import X, Y, CNOT, kron, chain, tag

xy = kron(X(0), Y(1))
tag(xy, "subblock")

composite_block = kron(xy, CNOT(3,4))
final_block = chain(composite_block, composite_block)

from qadence.draw import html_string # markdown-exec: hide
print(html_string(final_block)) # markdown-exec: hide
```

## Block execution

To quickly run quantum operations and access wavefunctions, samples or expectation values of
observables, one can use the convenience functions `run`, `sample` and `expectation`. The following
example shows an execution workflow with the natively available `PyQTorch` backend:

```python exec="on" source="material-block" result="json" session="index"
from qadence import chain, add, H, Z, run, sample, expectation

n_qubits = 2
block = chain(H(0), H(1))

# Compute the wavefunction.
# Please check the documentation for other available backends.
wf = run(block)
print(f"{wf = }") # markdown-exec: hide

# Sample the resulting wavefunction with a given number of shots.
xs = sample(block, n_shots=1000)
print(f"{xs = }") # markdown-exec: hide

# Compute an expectation based on an observable of Pauli-Z operators.
obs = add(Z(i) for i in range(n_qubits))
ex = expectation(block, obs)
print(f"{ex = }") # markdown-exec: hide
```

More fine-grained control and better performance is provided via the high-level `QuantumModel` abstraction.

## Execution via `QuantumCircuit` and `QuantumModel`

Quantum programs in Qadence are constructed in two steps:

1. Build a [`QuantumCircuit`][qadence.circuit.QuantumCircuit] which ties together a composite block and a register.
2. Define a [`QuantumModel`](/tutorials/quantummodels) which differentiates, compiles and executes the circuit.

`QuantumCircuit` is a central class in Qadence and circuits are abstract
objects from the actual hardware/simulator that they are expected to be executed on.
They require to specify the `Register` of resources to execute your program on. Previous examples
were already using `QuantumCircuit` with a `Register` that fits the qubit support for the given block.

```python exec="on" source="material-block" result="json"
from qadence import QuantumCircuit, Register, H, chain

# NOTE: Run a block which supports two qubits
# on a register of three qubits.
register = Register(3)
circuit = QuantumCircuit(register, chain(H(0), H(1)))
print(f"circuit = {circuit}") # markdown-exec: hide
```

!!! note "Registers and qubit supports"
    Registers can also be constructed from qubit coordinates to create arbitrary register
    topologies. See details in the [digital-analog](/digital_analog_qc/analog-basics.md) section.
	Qubit supports are subsets of the circuit register tied to blocks.


`QuantumModel` is another central class in Qadence. It specifies a [Backend](/tutorials/backend.md) for
the differentiation, compilation and execution of the abstract circuit.

```python exec="on" source="material-block" result="json"
from qadence import BackendName, DiffMode, QuantumCircuit, QuantumModel, Register, H, chain

reg = Register(3)
circ = QuantumCircuit(reg, chain(H(0), H(1)))
model = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

xs = model.sample(n_shots=100)
print(f"{xs = }") # markdown-exec: hide
```

For more details on `QuantumModel`, see [here](/tutorials/quantummodels).
