# State Conventions

Here is an overview of the state conventions used in Qadence together with practical examples.

## Qubit register order

Qubit registers in quantum computing are often indexed in increasing or decreasing order from left to right. In Qadence, the convention is qubit indexation in increasing order. For example, a register of four qubits in bra-ket notation reads:

$$|q_0, q_1, q_2, q_3\rangle$$

Furthermore, when displaying a quantum circuit, qubits are ordered from top to bottom.

## Basis state order

Basis state ordering refers to how basis states are ordered when considering the conversion from bra-ket notation to the standard linear algebra basis. In Qadence, basis states are ordered in the following manner:

$$
\begin{align}
|00\rangle = [1, 0, 0, 0]^T\\
|01\rangle = [0, 1, 0, 0]^T\\
|10\rangle = [0, 0, 1, 0]^T\\
|11\rangle = [0, 0, 0, 1]^T
\end{align}
$$

## Endianness

Endianness refers to the storage convention for binary information (in *bytes*) in a classical memory register. In quantum computing, information is either stored in bits or in qubits. The most commonly used conventions are:

- A **big-endian** system stores the **most significant bit** of a binary word at the smallest memory address.
- A **little-endian** system stores the **least significant bit** of a binary word at the smallest memory address.

Given the register convention in Qadence, the integer $2$ written in binary big-endian as $10$ can be encoded in a qubit register in both big-endian as $|10\rangle$ or little-endian as $|01\rangle$.

The convention for Qadence is **big-endian**.

## In practice

In practical scenarios, conventions regarding *register order*, *basis state order* and *endianness* are very much intertwined, and identical results can be obtained by fixing or varying any of them. In Qadence, we assume that qubit ordering and basis state ordering is fixed, and allow an `endianness` argument that can be passed to control the expected result. Here are a few examples:

### Quantum states

A simple and direct way to exemplify the endianness convention is the following:

```python exec="on" source="material-block" result="json" session="end-0"
import qadence as qd
from qadence import Endianness

# The state |10>, the 3rd basis state.
state_big = qd.product_state("10", endianness = Endianness.BIG) # or just "Big"

# The state |01>, the 2nd basis state.
state_little = qd.product_state("10", endianness = Endianness.LITTLE) # or just "Little"

print(state_big) # markdown-exec: hide
print(state_little) # markdown-exec: hide
```

Here, a bit word expressed as a Python string is used to create the respective basis state following both conventions. However, note that the same results can be obtained by fixing the endianness convention as big-endian (thus creating the state $|10\rangle$ in both cases), and changing the basis state ordering. A similar argument holds for fixing both endianness and basis state ordering and simply changing the qubit index order.

Another example where endianness directly comes into play is when *measuring* a register. A big or little endian measurement will choose the first or the last qubit, respectively, as the most significant bit. Let's see this in an example:

```python exec="on" source="material-block" result="json" session="end-0"
from qadence import I, H

# Create superposition state: |00> + |01> (normalized)
block = I(0) @ H(1)  # Identity on qubit 0, Hadamard on qubit 1

# Generate bitword samples following both conventions
# Samples "00" and "01"
result_big = qd.sample(block, endianness = Endianness.BIG)
# Samples "00" and "10"
result_little = qd.sample(block, endianness = Endianness.LITTLE)

print(result_big) # markdown-exec: hide
print(result_little) # markdown-exec: hide
```

In Qadence we can also invert endianness of many objects with the same `invert_endianness` function:

```python exec="on" source="material-block" result="json" session="end-0"
# Equivalent to sampling in little-endian.
print(qd.invert_endianness(result_big))

# Equivalent to a state created in little-endian
print(qd.invert_endianness(state_big))
```

### Quantum operations

When looking at quantum operations in matrix form, the usage of the term *endianness* slightly deviates from its absolute definition. To exemplify, we may consider the CNOT operation with `control = 0` and `target = 1`. This operation is often described with two different matrices:

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

The difference between these two matrices can be easily explained either by considering a different ordering of the qubit indices, or a different ordering of the basis states. In Qadence, both can be retrieved through the endianness argument:

```python exec="on" source="material-block" result="json" session="end-0"
matrix_big = qd.block_to_tensor(qd.CNOT(0, 1), endianness=Endianness.BIG)
print(matrix_big.detach())
print("") # markdown-exec: hide
matrix_little = qd.block_to_tensor(qd.CNOT(0, 1), endianness=Endianness.LITTLE)
print(matrix_little.detach())
```

## Backends

An important part of having clear state conventions is that we need to make sure our results are consistent accross different computational backends, which may have their own conventions that we need to take into account. In Qadence, we take care of this automatically, such that by calling a certain operation for different backends we expect a result that is equivalent in qubit ordering.

```python exec="on" source="material-block" result="json" session="end-0"
import warnings # markdown-exec: hide
warnings.filterwarnings("ignore") # markdown-exec: hide

import qadence as qd
from qadence import BackendName
import torch

# RX(pi/4) on qubit 1
n_qubits = 2
op = qd.RX(1, torch.pi/4)

print("Same sampling order:")
print(qd.sample(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PYQTORCH))
print(qd.sample(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.BRAKET))
print(qd.sample(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PULSER))
print("") # markdown-exec: hide
print("Same wavefunction order:")
print(qd.run(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PYQTORCH))
print(qd.run(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.BRAKET))
print(qd.run(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PULSER))
```
