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

Other standard operations are expressed as projectors in Qadence. For instance, the number operator $N=\dfrac{1}{2}(I-Z)=|1\rangle\langle 1|$ is used to build $\textrm{CNOT}(i,j)=N(i)\otimes(X(j)-I(j))$. In fact, `CNOT` can also be defined as a projector controlled-unitary operation: $\textrm{CNOT}(i,j)=|0\rangle\langle 0|\otimes \mathbb{I}+|1\rangle\langle 1|\otimes X$ and we can check their matrix representations are identical:

```python exec="on" source="material-block" session="projector" result="json"
from qadence import kron, I, X, CNOT

# Define a projector for |0> onto the qubit labelled 0.
projector0 = Projector(ket="0", bra="0", qubit_support=0)

# Define a projector for |1> onto the qubit labelled 0.
projector1 = Projector(ket="1", bra="1", qubit_support=0)

# Construct the projector controlled CNOT.
projector_cnot = kron(projector0, I(1)) + kron(projector1, X(1))

# Get the underlying unitary.
projector_cnot_matrix = block_to_tensor(projector_cnot)

# Qadence CNOT unitary.
qadence_cnot_matrix = block_to_tensor(CNOT(0,1))

print(f"projector cnot matrix = {projector_cnot_matrix}") # markdown-exec: hide
print(f"qadence cnot matrix = {qadence_cnot_matrix}") # markdown-exec: hide
```

!!! warning
    Projectors lead to non-unitary computations only supported by the PyQTorch backend.

```python exec="on" source="material-block" session="projector" result="json"
from qadence import chain, run
from qadence.operations import H, CNOT

# Define a projector for |1> onto the qubit labelled 1.
projector_block = Projector(ket="1", bra="1", qubit_support=1)

# Some non unitary computation.
non_unitary_block = chain(H(0), CNOT(0,1), projector_block)

projected_wf = run(non_unitary_block)  # Run on PyQTorch.

print(f"projected_wf = {projected_wf}") # markdown-exec: hide
```
