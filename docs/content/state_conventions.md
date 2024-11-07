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

## Quantum states

In practical scenarios, conventions regarding *register order*, *basis state order* and *endianness* are very much intertwined, and identical results can be obtained by fixing or varying any of them. In Qadence, we assume that qubit ordering and basis state ordering is fixed, and allow an `endianness` argument that can be passed to control the expected result. Here are a few examples:

A simple and direct way to exemplify the endianness convention is using convenience functions for state preparation.

!!! note "Bitstring convention as inputs"
	When a bitstring is passed as input to a function for state preparation, it has to be understood in
	**big-endian** convention.

```python exec="on" source="material-block" result="json" session="end-0"
from qadence import Endianness, product_state

# The state |10>, the 3rd basis state.
state_big = product_state("10", endianness=Endianness.BIG) # or just "Big"

# The state |01>, the 2nd basis state.
state_little = product_state("10", endianness=Endianness.LITTLE) # or just "Little"

print(f"State in big endian = {state_big}") # markdown-exec: hide
print(f"State in little endian = {state_little}") # markdown-exec: hide
```

Here, a bitword expressed as a Python string to encode the integer 2 in big-endian is used to create the respective basis state in both conventions. However, note that the same results can be obtained by fixing the endianness convention as big-endian (thus creating the state $|10\rangle$ in both cases), and changing the basis state ordering. A similar argument holds for fixing both endianness and basis state ordering and simply changing the qubit index order.

Another example where endianness directly comes into play is when *measuring* a register. A big- or little-endian measurement will choose the first or the last qubit, respectively, as the most significant bit. Let's see this in an example:

```python exec="on" source="material-block" result="json" session="end-0"
from qadence import I, H, sample

# Create superposition state: |00> + |01> (normalized)
block = I(0) @ H(1)  # Identity on qubit 0, Hadamard on qubit 1

# Generate bitword samples following both conventions
# Samples "00" and "01"
result_big = sample(block, endianness=Endianness.BIG)
# Samples "00" and "10"
result_little = sample(block, endianness=Endianness.LITTLE)

print(f"Sample in big endian = {result_big}") # markdown-exec: hide
print(f"Sample in little endian = {result_little}") # markdown-exec: hide
```

In Qadence, endianness can be flipped for many relevant objects:

```python exec="on" source="material-block" result="json" session="end-0"
from qadence import invert_endianness

# Equivalent to sampling in little-endian.
flip_big_sample = invert_endianness(result_big)
print(f"Flipped sample = {flip_big_sample}") # markdown-exec: hide

# Equivalent to a state created in little-endian.
flip_big_state = invert_endianness(state_big)
print(f"Flipped state = {flip_big_state}") # markdown-exec: hide
```

## Quantum operations

When looking at the matricial form of quantum operations, the usage of the term *endianness* becomes slightly abusive. To exemplify, we may consider the `CNOT` operation with `control = 0` and `target = 1`. This operation is often described with two different matrices:

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

The difference can be easily explained either by considering a different ordering of the qubit indices, or a different ordering of the basis states. In Qadence, both can be retrieved through the `endianness` argument:

```python exec="on" source="material-block" result="json" session="end-0"
from qadence import block_to_tensor, CNOT

matrix_big = block_to_tensor(CNOT(0, 1), endianness=Endianness.BIG)
print("CNOT matrix in big endian =\n") # markdown-exec: hide
print(f"{matrix_big.detach()}\n") # markdown-exec: hide
matrix_little = block_to_tensor(CNOT(0, 1), endianness=Endianness.LITTLE)
print("CNOT matrix in little endian =\n") # markdown-exec: hide
print(f"{matrix_little.detach()}") # markdown-exec: hide
```

## Backends

An important part of having clear state conventions is that we need to make sure our results are consistent accross different computational backends, which may have their own conventions. In Qadence, this is taken care of automatically: by calling operations for different backends, the result is expected to be equivalent up to qubit ordering.

```python exec="on" source="material-block" result="json" session="end-0"
import warnings # markdown-exec: hide
warnings.filterwarnings("ignore") # markdown-exec: hide
from qadence import BackendName, RX, run, sample, PI

# RX(PI/4) on qubit 1
n_qubits = 2
op = RX(1, PI/4)

print("Same sampling order in big endian:\n") # markdown-exec: hide
print(f"On PyQTorch = {sample(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PYQTORCH)}") # markdown-exec: hide
print(f"On Pulser = {sample(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PULSER)}\n") # markdown-exec: hide
print("Same wavefunction order:\n") # markdown-exec: hide
print(f"On PyQTorch = {run(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PYQTORCH)}") # markdown-exec: hide
print(f"On Pulser = {run(n_qubits, op, endianness=Endianness.BIG, backend=BackendName.PULSER)}") # markdown-exec: hide
```
