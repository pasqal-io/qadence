This section introduces the `ProjectorBlock` as an implementation for the the quantum mechanical projection operation onto the subspace spanned by $|a\rangle$: $\mathbb{\hat{P}}=|a\rangle \langle a|$. It evaluates the outer product for bras and kets expressed as bitstrings for a given qubit support. They have to possess matching lengths.

```python exec="on" source="material-block" session="projector" result="json"
from qadence.blocks import block_to_tensor
from qadence.operations import Projector  # Projector as an operation.

# Define a projector for |1> onto the qubit labelled 0.
projector_block = Projector(ket="1", bra="1", qubit_support=0)

# As any block, the matrix representation can be retrieved.
projector_matrix = block_to_tensor(projector_block)

print(f"projector matrix = {projector_matrix}") # markdown-exec: hide
```

Other standard operations are expressed as projectors in Qadence. For instance, the number operator $N=\dfrac{1}{2}(I-Z)=|1\rangle\langle 1|$ is used to build $\textrm{CNOT}(i,j)=N(i)\otimes(X(j)-I(j))$. In fact, `CNOT` can also be defined as a projector controlled-unitary operation: $\textrm{CNOT}(i,j)=|0\rangle\langle 0|_i\otimes \mathbb{I}(j)+|1\rangle\langle 1|_i\otimes X(j)$ and we can check their matrix representations are identical:

```python exec="on" source="material-block" session="projector" result="json"
from qadence.blocks import block_to_tensor
from qadence import kron, I, X, CNOT

# Define a projector for |0> onto the qubit labelled 0.
projector0 = Projector(ket="0", bra="0", qubit_support=0)

# Define a projector for |1> onto the qubit labelled 0.
projector1 = Projector(ket="1", bra="1", qubit_support=0)

# Construct the projector controlled CNOT.
projector_cnot = kron(projector0, I(1)) + kron(projector1, X(1))

# Get the underlying unitary.
projector_cnot_matrix = block_to_tensor(projector_cnot)

# Qadence CNOT unitary defined as N @ (X-I).
qadence_cnot_matrix = block_to_tensor(CNOT(0,1))

print(f"projector cnot matrix = {projector_cnot_matrix}") # markdown-exec: hide
print(f"qadence cnot matrix = {qadence_cnot_matrix}") # markdown-exec: hide
```

Another example is the canonical SWAP gate that can be defined as $SWAP(i,j)=|00\rangle\langle 00|+|01\rangle\langle 10|+|10\rangle\langle 01|+|11\rangle\langle 11|$. Indeed, it can be shown that their matricial representations are again identical:

```python exec="on" source="material-block" session="projector" result="json"
from qadence.blocks import block_to_tensor
from qadence import SWAP

# Define all projectors.
projector00 = Projector(ket="00", bra="00", qubit_support=(0, 1))
projector01 = Projector(ket="01", bra="10", qubit_support=(0, 1))
projector10 = Projector(ket="10", bra="01", qubit_support=(0, 1))
projector11 = Projector(ket="11", bra="11", qubit_support=(0, 1))

# Construct the SWAP gate.
projector_swap = projector00 + projector10 + projector01 + projector11

# Get the underlying unitary.
projector_swap_matrix = block_to_tensor(projector_swap)

# Qadence SWAP unitary.
qadence_swap_matrix = block_to_tensor(SWAP(0,1))

print(f"projector swap matrix = {projector_swap_matrix}") # markdown-exec: hide
print(f"qadence swap matrix = {qadence_swap_matrix}") # markdown-exec: hide
```

!!! warning
    Projectors lead to non-unitary computations only supported by the PyQTorch backend.


To examplify this point, let's run some non-unitary computation involving projectors.

```python exec="on" source="material-block" session="projector" result="json"
from qadence import chain, run
from qadence.operations import H, CNOT

# Define a projector for |1> onto the qubit labelled 1.
projector_block = Projector(ket="1", bra="1", qubit_support=1)

# Some non-unitary computation.
non_unitary_block = chain(H(0), CNOT(0,1), projector_block)

# Projected wavefunction becomes unnormalized
projected_wf = run(non_unitary_block)  # Run on PyQTorch.

print(f"projected_wf = {projected_wf}") # markdown-exec: hide
```
