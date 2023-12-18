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

Other standard operations are expressed as projectors in Qadence. For instance, the number operator $N=\dfrac{1}{2}(I-Z)=|1\rangle\langle 1|$ is used to build projector controlled-unitary gates: $\textrm{CNOT}(i,j)=N(i)\otimes(X(j)-I(j))$.

!!! warning
    Projectors lead to non-unitary computations only supported by the PyQTorch backend.

```python exec="on" source="material-block" session="projector" result="json"
from qadence import chain, run
from qadence.operations import H, CNOT

# Define a projector for |1> onto the qubit labelled 1.
projector_block = Projector(ket="1", bra="1", qubit_support=1)

# Some non unitary computation.
non_unitary_block = chain(H(0), CNOT(0,1), projector_block)

non_unitary_wf = run(non_unitary_block)  # Run on PyQTorch.

print(f"non_unitary_wf = {non_unitary_wf}") # markdown-exec: hide
```
