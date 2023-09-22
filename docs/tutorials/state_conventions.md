# State Conventions

Here we describe the state conventions used in `qadence` and give a few practical examples.

## Qubit register order

Qubit registers in quantum computing are often indexed in increasing or decreasing order. In `qadence` we use an increasing order. For example, for a register of 4 qubits we have:

$$q_0 \otimes q_1 \otimes q_2 \otimes q_3$$

Or alternatively in bra-ket notation,

$$|q_0, q_1, q_2, q_3\rangle$$

Furthermore, when displaying a quantum circuit, the qubits are ordered from top to bottom.

## Basis state order

Basis state ordering refers to how basis states are ordered when considering the conversion from bra-ket notation to the standard linear algebra basis. In `qadence` the basis states are ordered in the following manner:

$$
\begin{align}
|00\rangle = [1, 0, 0, 0]^T\\
|01\rangle = [0, 1, 0, 0]^T\\
|10\rangle = [0, 0, 1, 0]^T\\
|11\rangle = [0, 0, 0, 1]^T
\end{align}
$$

## Endianness

Endianness refers to the convention of how binary information is stored in a memory register. Tyically, in classical computers, it refers to the storage of *bytes*. However, in quantum computing information is mostly described in terms of single bits, or qubits. The most commonly used conventions are:

- A **big-endian** system stores the **most significant bit** of a word at the smallest memory address.
- A **little-endian** system stores the **least significant bit** of a word at the smallest memory address.

Given the register convention described for `qadence`, as an example, the integer $2$ written in binary as $10$ can be encoded in a qubit register in both big-endian as $|10\rangle$ or little-endian as $|01\rangle$.

In general, the default convention for `qadence` is **big-endian**.

## In practice

In practical scenarios, the conventions regarding *register order*, *basis state order* and *endianness* are very much connected, and the same results can be obtained by fixing or varying any of them. In `qadence`, we assume that qubit ordering and basis state ordering is fixed, and allow an `endianness` argument that can be passed to control the expected result. We now describe a few examples:

### Quantum states

A simple and direct way to exemplify the endianness convention is the following:

```python exec="on" source="material-block" result="json" session="end-0"
import qadence as qd

state_big = qd.product_state("10", endianness = qd.Endianness.BIG) # or just "Big"
state_little = qd.product_state("10", endianness = qd.Endianness.LITTLE) # or just "Little"

print(state_big) # The state |10>, the 3rd basis state.
print(state_little) # The state |01>, the 2nd basis state.
```

Here we took a bit word written as a Python string and used it to create the respective basis state following both conventions. However, note that we would actually get the same results by saying that we fixed the endianness convention as big-endian, thus creating the state $|10\rangle$ in both cases, but changed the basis state ordering. We could also make a similar argument for fixing both endianness and basis state ordering and simply changing the qubit index order. This is simply an illustration of how these concepts are connected.

Another example where endianness will come directly into play is when *measuring* a register. A big or little endian measurement will choose the first or the last qubit, respectively, as the most significant bit. Let's see this in an example:

```python exec="on" source="material-block" result="json" session="end-0"
# Create superposition state: |00> + |01> (normalized)
block = qd.I(0) @ qd.H(1)  # Identity on qubit 0, Hadamard on qubit 1

# Generate bitword samples following both conventions
result_big = qd.sample(block, endianness = qd.Endianness.BIG)
result_little = qd.sample(block, endianness = qd.Endianness.LITTLE)

print(result_big) # Samples "00" and "01"
print(result_little) # Samples "00" and "10"
```

In `qadence` we can also invert endianness of many objects with the same `invert_endianness` function:

```python exec="on" source="material-block" result="json" session="end-0"
# Equivalent to sampling in little-endian.
print(qd.invert_endianness(result_big))

# Equivalent to a state created in little-endian
print(qd.invert_endianness(state_big))
```

### Quantum operations

When looking at quantum operations in matrix form, our usage of the term *endianness* slightly deviates from its absolute definition. To exemplify, we maybe consider the CNOT operation with `control = 0` and `target = 1`. This operation is often described with two different matrices:

$$
\text{CNOT(0, 1)} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
\qquad
\text{or}
\qquad
\text{CNOT(0, 1)} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
\end{bmatrix}
$$

The difference between these two matrices can be easily explained either by considering a different ordering of the qubit indices, or a different ordering of the basis states. In `qadence`, we can get both through the endianness argument:

```python exec="on" source="material-block" result="json" session="end-0"
matrix_big = qd.block_to_tensor(qd.CNOT(0, 1), endianness = "Big")
print(matrix_big.detach())
print("") # markdown-exec: hide
matrix_big = qd.block_to_tensor(qd.CNOT(0, 1), endianness = "Little")
print(matrix_big.detach())
```

While the usage of the term here may not be fully accurate, it helps with keeping a consistent interface, and it still relates to the same general idea of qubit index ordering or which qubit is considered the most significant.

## Backends

An important part of having clear state conventions is that we need to make sure our results are consistent accross different computational backends, which may have their own conventions that we need to take into account. In `qadence` we take care of this automatically, such that by calling a certain operation for different backends we expect a result that is equivalent in qubit ordering.

```python exec="on" source="material-block" result="json" session="end-0"
import warnings # markdown-exec: hide
warnings.filterwarnings("ignore") # markdown-exec: hide

import qadence as qd
import torch

# RX(pi/4) on qubit 1
n_qubits = 2
op = qd.RX(1, torch.pi/4)

print("Same sampling order:")
print(qd.sample(n_qubits, op, endianness = "Big", backend = qd.BackendName.PYQTORCH))
print(qd.sample(n_qubits, op, endianness = "Big" ,backend = qd.BackendName.BRAKET))
print(qd.sample(n_qubits, op, endianness = "Big", backend = qd.BackendName.PULSER))
print("") # markdown-exec: hide
print("Same wavefunction order:")
print(qd.run(n_qubits, op, endianness = "Big", backend = qd.BackendName.PYQTORCH))
print(qd.run(n_qubits, op, endianness = "Big" ,backend = qd.BackendName.BRAKET))
print(qd.run(n_qubits, op, endianness = "Big", backend = qd.BackendName.PULSER))
```
