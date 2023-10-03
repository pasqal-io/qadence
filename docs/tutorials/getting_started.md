Quantum programs in Qadence are constructed via a block-system, which makes it easily possible to
compose small, *primitive* blocks to obtain larger, *composite* blocks.  This approach is very
different from how other frameworks (like Qiskit) construct circuits which follow an object-oriented
approach.

## [`PrimitiveBlock`][qadence.blocks.primitive.PrimitiveBlock]

A `PrimitiveBlock` is a basic operation such as a digital gate or an analog
time-evolution block. This is the only concrete element of the block system
and the program can always be decomposed into a list of `PrimitiveBlock`s.

Two examples of primitive blocks are the `X` and the `CNOT` gates:

```python exec="on" source="material-block" html="1"
from qadence import RX

# a rotation gate on qubit 0
rx0 = RX(0, 0.5)
from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(rx0), size="2,2")) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1"
from qadence import CNOT

# a CNOT gate with control=0 and target=1
c01 = CNOT(0, 1)
from qadence.draw import html_string # markdown-exec: hide
from qadence import chain # markdown-exec: hide
print(html_string(chain(c01), size="2,2")) # markdown-exec: hide
```

You can find a list of all instances of primitive blocks (also referred to as *operations*)
[here](/qadence/operations.md).


## [`CompositeBlock`][qadence.blocks.composite.CompositeBlock]

Larger programs can be constructed from three operations:
[`chain`][qadence.blocks.utils.chain],
[`kron`][qadence.blocks.utils.kron], and
[`add`][qadence.blocks.utils.add].

[**`chain`**][qadence.blocks.utils.chain]ing blocks applies a set of sub-blocks in series, i.e. one
after the other on the *same or different qubit support*. A `ChainBlock` is akin to applying a
matrix product of the sub-blocks which is why it can also be used via the `*`-operator.
```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, chain

i = chain(X(0), X(0))
from qadence.draw import html_string # markdown-exec: hide
print(html_string(i, size="2,2")) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1" session="i-xx"
xx = X(0) * X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(xx, size="2,2")) # markdown-exec: hide
```

??? note "Get the matrix of a block"
    You can always translate a block to its matrix representation.  Note that the returned tensor
    contains a batch dimension because of parametric blocks.
    ```python exec="on" source="material-block" result="json" session="i-xx"
    print("X(0) * X(0)")
    print(i.tensor())
    print("\n") # markdown-exec: hide
    print("X(0) * X(1)")
    print(xx.tensor())
    ```

In order to stack blocks (i.e. apply them simultaneously) you can use
[**`kron`**][qadence.blocks.utils.kron].  A `KronBlock` applies a set of sub-blocks simultaneously on
*different qubit support*. This is akin to applying a tensor product of the sub-blocks.
```python exec="on" source="material-block" html="1" session="i-xx"
from qadence import X, kron

xx = kron(X(0), X(1))
# equivalent to X(0) @ X(1)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(xx, size="2,2")) # markdown-exec: hide
```
"But this is the same as `chain`ing!", you may say. And yes, for the digital case `kron` and `chain`
have the same meaning apart from how they influence the plot of your block. However, Qadence also
supports *analog* blocks, which need this concept of sequential/simultaneous blocks. To learn more
about analog blocks check the [digital-analog](/digital_analog_qc/analog-basics) section.

Finally, we have [**`add`**][qadence.blocks.utils.add]. This simply sums the corresponding matrix of
each sub-block.  `AddBlock`'s can also be used to construct Pauli operators.

!!! warning
    Notice that `AddBlock`s can give rise to non-unitary blocks and thus might not be
    executed by all backends but only by certain simulators.

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

### Quick, one-off execution
To quickly run quantum operations and access wavefunctions, samples or expectation values of
observables, one can use the convenience functions `run`, `sample` and `expectation`.
More fine-grained control and better performance is provided via the `QuantumModel`.

??? note "The quick and dirty way"
    Define a simple quantum program and perform some quantum operations on it:
    ```python exec="on" source="material-block" result="json" session="index"
    from qadence import chain, add, H, Z, run, sample, expectation

    n_qubits = 2
    block = chain(H(0), H(1))

    # compute wavefunction with the `pyqtorch` backend
    # check the documentation for other available backends!
    wf = run(block)
    print(f"{wf = }") # markdown-exec: hide

    # sample the resulting wavefunction with a given number of shots
    xs = sample(block, n_shots=1000)
    print(f"{xs = }") # markdown-exec: hide

    # compute an expectation based on an observable
    obs = add(Z(i) for i in range(n_qubits))
    ex = expectation(block, obs)
    print(f"{ex = }") # markdown-exec: hide
    ```

### Proper execution via `QuantumCircuit` and `QuantumModel`

Quantum programs in qadence are constructed in two steps:

1. Define a `QuantumCircuit` which ties together a block and a register to a well-defined circuit.
2. Define a `QuantumModel` which takes care of compiling and executing the circuit.

#### 1. [`QuantumCircuit`][qadence.circuit.QuantumCircuit]s

The `QuantumCircuit` is one of the central classes in Qadence. For example, to specify the `Register`
to run your block on you use a `QuantumCircuit` (under the hood the functions above were already
using `QuantumCircuits` with a `Register` that fits the qubit support of the given block).

The `QuantumCircuit` ties a block together with a register.

```python exec="on" source="material-block" result="json"
from qadence import QuantumCircuit, Register, H, chain

# NOTE: we run a block which supports two qubits
# on a register with three qubits
reg = Register(3)
circ = QuantumCircuit(reg, chain(H(0), H(1)))
print(circ) # markdown-exec: hide
```

!!! note "`Register`s"
    Registers can also be constructed e.g. from qubit coordinates to create arbitrary register
    layouts, but more on that in the [digital-analog](/digital_analog_qc/analog-basics.md) section.


#### 2. [`QuantumModel`](/tutorials/quantumodels)s

`QuantumModel`s are another central class in Qadence's library. Blocks and circuits are completely abstract
objects that have nothing to do with the actual hardware/simulator that they are running on. This is
where the `QuantumModel` comes in. It contains a [`Backend`](/tutorials/backend.md) and a
compiled version of your abstract circuit (constructed by the backend).

The `QuantumModel` is also what makes our circuit *differentiable* (either via automatic
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

    Qadence offers some convenience routines for preparing the initial quantum state.
    These routines are divided into two approaches:
    * generate the initial state as a dense matrix (routines with `_state` postfix).
      This only works for backends which support state vectors as inputs, currently
      only PyQ.
    * generate the initial state from a suitable quantum circuit (routines with
      `_block` postfix). This is available for every backend and it should be added
      in front of the desired quantum circuit to simulate.

    Let's illustrate the usage of the state preparation routine. For more details,
    please refer to the [API reference](/qadence/index).

    ```python exec="on" source="material-block" result="json" session="seralize"
    from qadence import random_state, product_state, is_normalized, StateGeneratorType

    # random initial state
    # the default `type` is StateGeneratorType.HaarMeasureFast
    state = random_state(n_qubits=2, type=StateGeneratorType.RANDOM_ROTATIONS)
    print(f"Random initial state generated with rotations:\n {state.detach().numpy().flatten()}")

    # check the normalization
    assert is_normalized(state)

    # product state from a given bitstring
    # remember that qadence follows the big endian convention
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
