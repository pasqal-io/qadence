This section introduces the `ProjectorBlock` as an implementation for the the quantum mechanical projection operation. It evaluates to the outer product of a ket and a bra expressed as bitstrings.

!!! warning
    Projectors lead to non-unitary computations.


```python exec="on" source="material-block" session="noise" result="json"
from qadence.blocks import block_to_tensor
from qadence.operations import Projector  # Projector as an operation.


projector_block = Projector(ket="1", bra="1", qubit_support=0)

# As any block, the matrix representation can be retrieved.
projector_matrix = block_to_tensor(projector_block)

print(f"projector matrix = {projector_matrix}") # markdown-exec: hide
```

Other standard operations can be expressed as projectors: for instance, the number operator $N=\dfrac{1}{2}(I-Z)=|1\rangle\langle 1|$ is used for projector controlled-unitary gates: $\textrm{CNOT}(i,j)=N(i)\otimes(X(j)-I(j))$
