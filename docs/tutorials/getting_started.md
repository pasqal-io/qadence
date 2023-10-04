Quantum programs in Qadence are constructed via a block-system, with an emphasis on composability of
*primitive* blocks to obtain larger, *composite* blocks. This functional approach is different from other frameworks
which follow a more object-oriented way to construct circuits and express programs.

## Primitive Blocks

A [`PrimitiveBlock`][qadence.blocks.primitive.PrimitiveBlock] represents a digital or an analog time-evolution quantum operation applied to a qubit support.
Programs can always be decomposed down into a sequence of `PrimitiveBlock` elements.

Two canonical examples of primitive blocks are the parametrized `RX` and the `CNOT` gates:

```python exec="on" source="material-block" html="1"
from qadence import RX

# A rotation gate on qubit 0 with a fixed numerical parameter.
rx0 = RX(0, 0.5)
from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(rx0), size="2,2")) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1"
from qadence import CNOT

# A CNOT gate with control on qubit 0 and target on qubit 1.
c01 = CNOT(0, 1)
from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(c01), size="2,2")) # markdown-exec: hide
```

You can find a list of all instances of primitive blocks (also referred to as *operations*)
[here](/qadence/operations.md).


## Composite Blocks

Programs can be expressed by composing blocks to result in a [`CompositeBlock`][qadence.blocks.composite.CompositeBlock] using three fundamental operations:
[_chain_][qadence.blocks.utils.chain],
[_kron_][qadence.blocks.utils.kron], and
[_add_][qadence.blocks.utils.add].

- [**chain**][qadence.blocks.utils.chain] applies a set of blocks in sequence, on the *same or overlapping qubit supports* and results in `ChainBlock`.
It is akin to applying a matrix product of the sub-blocks with the `*` operator.

```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, chain

# Chaining on the same qubit.
chain_x = chain(X(0), X(0))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(chain_x, size="2,2")) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1" session="i-xx"
# Chaining on different qubits. Identical to the kron operation.
chain_xx = X(0) * X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(chain_xx, size="2,2")) # markdown-exec: hide
```

??? note "Get the matrix of a block"
    You can always translate a block to its matrix representation.  Note that the returned tensor
    contains a batch dimension because of parametric blocks.
    ```python exec="on" source="material-block" result="json" session="i-xx"
    print("X(0) * X(0)")
    print(chain_x.tensor())
    print("\n") # markdown-exec: hide
    print("X(0) * X(1)")
    print(chain_xx.tensor())
    ```

- [**kron**][qadence.blocks.utils.kron] applies a set of blocks in parallel (simultaneously) on *disjoint qubit support* and results in a `KronBlock`. This is akin to applying a tensor product of the sub-blocks with the `@` operator.
```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, kron

kron_xx = kron(X(0), X(1))
# Equivalent to X(0) @ X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(kron_xx, size="2,2")) # markdown-exec: hide
```

!!! warning The next section is rather unclear.


"But this is the same as `chain`ing!", you may say. And yes, for the digital case `kron` and `chain`
have the same meaning apart from how they influence the plot of your block. However, Qadence also
supports *analog* blocks, which need this concept of sequential/simultaneous blocks. To learn more
about analog blocks check the [digital-analog](/digital_analog_qc/analog-basics) section.

- [**add**][qadence.blocks.utils.add] sums the corresponding matrix of
each sub-block and results in a `AddBlock`. It can be used to construct Pauli operators.

!!! warning
    Notice that `AddBlock`s can give rise to non-unitary blocks and thus might not be
    executable on all backends but only by certain simulators.

```python exec="on" source="material-block" result="json"
from qadence import X, Z

xz = X(0) + Z(0)
print(xz.tensor())
```

Finally, a slightly more complicated example.
```python exec="on" source="material-block" html="1" session="getting_started"
from qadence import X, Y, CNOT, kron, chain, tag

xy = chain(X(0), Y(1))
tag(xy, "subblock")

composite_block = kron(xy, CNOT(3,4))
final_block = chain(composite_block, composite_block)

# tag the block with a human-readable name
tag(final_block, "my_block")
from qadence.draw import html_string # markdown-exec: hide
print(html_string(final_block, size="4,4")) # markdown-exec: hide
```

## Program execution

### Fast execution

To quickly run quantum operations and access wavefunctions, samples or expectation values of
observables, one can use the convenience functions `run`, `sample` and `expectation`.
More fine-grained control and better performance is provided via the `QuantumModel`.

??? note "Quick execution with the available PyQTorch backend"
    Define and execute a simple quantum program using convenience functions:

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

### Execution via `QuantumCircuit` and `QuantumModel`

Quantum programs in Qadence are constructed in two steps:

1. Define a `QuantumCircuit` which ties together a block and a register to a well-defined circuit.
2. Define a `QuantumModel` which compiles and execute the circuit.

#### 1. [`QuantumCircuit`][qadence.circuit.QuantumCircuit]s

`QuantumCircuit` is a central class in Qadence and circuits are abstract
objects from the actual hardware/simulator that they are expected to be executed on.
They require to specify the `Register` of resources to execute your program on. Function examples above
were already using `QuantumCircuit` with a `Register` that fits the qubit support of the given block.


```python exec="on" source="material-block" result="json"
from qadence import QuantumCircuit, Register, H, chain

# NOTE: We run a block which supports two qubits
# on a register of three qubits.
reg = Register(3)
circ = QuantumCircuit(reg, chain(H(0), H(1)))
print(circ) # markdown-exec: hide
```

!!! note Registers and qubit supports
    Registers can also be constructed e.g. from qubit coordinates to create arbitrary register
    topologies. See details in the [digital-analog](/digital_analog_qc/analog-basics.md) section.\n
	Qubit supports are tied to blocks and are a subset of the circuit register.


#### 2. [`QuantumModel`](/tutorials/quantumodels)s

`QuantumModel` is another central class in Qadence's library. It specifies a [Backend](/tutorials/backend.md) for
the execution and a compiled version of the abstract circuit.

The `QuantumModel` also provides circuit *differentiablity* features (either via automatic
differentiation, or on hardware via parameter shift rule).

```python exec="on" source="material-block" result="json"
from qadence import QuantumCircuit, QuantumModel, Register, H, chain

reg = Register(3)
circ = QuantumCircuit(reg, chain(H(0), H(1)))
model = QuantumModel(circ, backend="pyqtorch", diff_mode='ad')

xs = model.sample(n_shots=100)
print(f"{xs = }")
```

For more details on how to use `QuantumModel`s, see [here](/tutorials/quantummodels).


## State initialization

!!! warning "moved here from another page; improve?"
    #### Quantum state preparation

    Qadence offers convenience routines for preparing initial quantum states.
    These routines are divided into two approaches:

    - As a dense matrix.
    - From a suitable quantum circuit. This is available for every backend and it should be added
      in front of the desired quantum circuit to simulate.

    Let's illustrate the usage of the state preparation routine. For more details,
    please refer to the [API reference](/qadence/index).

    ```python exec="on" source="material-block" result="json" session="seralize"
    from qadence import random_state, product_state, is_normalized, StateGeneratorType

    # Random initial state.
    # the default `type` is StateGeneratorType.HaarMeasureFast
    state = random_state(n_qubits=2, type=StateGeneratorType.RANDOM_ROTATIONS)
    print(f"Random initial state generated with rotations:\n {state.detach().numpy().flatten()}")

    # Check the normalization.
    assert is_normalized(state)

    # Product state from a given bitstring.
    # NB: Qadence follows the big endian convention.
    state = product_state("01")
    print(f"Product state corresponding to bitstring '10':\n {state.detach().numpy().flatten()}")
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

    display(qc_with_state_prep)
    ```
