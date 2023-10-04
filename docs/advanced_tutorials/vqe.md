## Restricted Hamiltonian

Simple implementation of the UCC ansatz for computing the ground state of the H2
molecule. The Hamiltonian coefficients are taken from the following paper:
https://arxiv.org/pdf/1512.06860.pdf.

Simple 2 qubits unitary coupled cluster ansatz for H2 molecule
```python exec="on" source="material-block" html="1" session="vqe"
import torch
from qadence import X, RX, RY, RZ, CNOT, chain, kron

def UCC_ansatz_H2():
    ansatz=chain(
        kron(chain(X(0), RX(0, -torch.pi/2)), RY(1, torch.pi/2)),
        CNOT(1,0),
        RZ(0, f"theta"),
        CNOT(1,0),
        kron(RX(0, torch.pi/2), RY(1, -torch.pi/2))
    )
    return ansatz


from qadence.draw import html_string # markdown-exec: hide
print(html_string(UCC_ansatz_H2())) # markdown-exec: hide
```


Let's define the Hamiltonian of the problem in the following form: hamilt =
[list of coefficients, list of Pauli operators, list of qubits]. For example:
`hamilt=[[3,4],[[X,X],[Y]],[[0,1],[3]]]`.

In the following function we generate the Hamiltonian with the format above.

```python exec="on" source="material-block" html="1" session="vqe"
from typing import Iterable
from qadence import X, Y, Z, I, add
def make_hamiltonian(hamilt: Iterable, nqubits: int):

    nb_terms = len(hamilt[0])
    blocks = []

    for iter in range(nb_terms):
        block = kron(gate(qubit) for gate,qubit in zip(hamilt[1][iter], hamilt[2][iter]))
        blocks.append(hamilt[0][iter] * block)

    return add(*blocks)


nqbits = 2

# Hamiltonian definition using the convention outlined above
hamilt_R07 = [
    [0.2976, 0.3593, -0.4826,0.5818, 0.0896, 0.0896],
    [[I,I],[Z],[Z],[Z,Z],[X,X],[Y,Y]],
    [[0,1],[0],[1],[0,1],[0,1],[0,1]]
]

hamiltonian = make_hamiltonian(hamilt_R07, nqbits)
from qadence.draw import html_string # markdown-exec: hide
print(html_string(hamiltonian)) # markdown-exec: hide
```

Let's now create a `QuantumCircuit` representing the variational ansatz and plug
it into a `QuantumModel` instance. From there, it is very easy to compute the
energy by simply evaluating the expectation value of the Hamiltonian operator.

```python exec="on" source="material-block" result="json" session="vqe"
from qadence import QuantumCircuit, QuantumModel

ansatz = QuantumCircuit(nqbits, UCC_ansatz_H2())
model = QuantumModel(ansatz, observable=hamiltonian, backend="pyqtorch", diff_mode="ad")

values={}
out = model.expectation(values)
print(out)
```
Let's now resent the parameters and set them randomly before starting the optimization loop.

```python exec="on" source="material-block" result="json" session="vqe"
init_params = torch.rand(model.num_vparams)
model.reset_vparams(init_params)

n_epochs = 100
lr = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(n_epochs):
    optimizer.zero_grad()
    out=model.expectation({})
    out.backward()
    optimizer.step()

print("Ground state energy =", out.item(), "Hatree")
```

## Unrestricted Hamiltonian

This result is in line with what obtained in the reference paper. Let's now
perform the same calculations but with a standard hardware efficient ansatz
(i.e. not specifically tailored for the H2 molecule) and with an unrestricted
Hamiltonian on 4 qubits. The values of the coefficients are taken from BK Hamiltonian, page 28[^2].

```python exec="on" source="material-block" html="1" session="vqe"
from qadence import hea

nqbits = 4

gates = [[I,I,I,I],[Z],[Z],[Z],[Z,Z],[Z,Z],[Z,Z],[X,Z,X],[Y,Z,Y],[Z,Z,Z],[Z,Z,Z],[Z,Z,Z],[Z,X,Z,X],[Z,Y,Z,Y],[Z,Z,Z,Z]]
qubits = [[0,1,2,3],[0],[1],[2],[0,1],[0,2],[1,3],[2,1,0],[2,1,0],[2,1,0],[3,2,0],[3,2,1],[3,2,1,0],[3,2,1,0],[3,2,1,0]]
coeffs = [
    -0.81261,0.171201,0.16862325,- 0.2227965,0.171201,0.12054625,0.17434925  ,0.04532175,0.04532175,0.165868 ,
    0.12054625,-0.2227965 ,0.04532175 ,0.04532175,0.165868
]

hamilt_R074_bis = [coeffs,gates,qubits]
Hamiltonian_bis = make_hamiltonian(hamilt_R074_bis, nqbits)
ansatz_bis = QuantumCircuit(4, hea(nqbits))

from qadence.draw import html_string # markdown-exec: hide
print(html_string(ansatz_bis)) # markdown-exec: hide
```
```python exec="on" source="material-block" result="json" session="vqe"
model = QuantumModel(ansatz_bis, observable=Hamiltonian_bis, backend="pyqtorch", diff_mode="ad")

values={}
out=model.expectation(values)

# initialize some random initial parameters
init_params = torch.rand(model.num_vparams)
model.reset_vparams(init_params)

n_epochs = 100
lr = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(n_epochs):

    optimizer.zero_grad()
    out=model.expectation(values)
    out.backward()
    optimizer.step()
    if (i+1) % 10 == 0:
        print(f"Epoch {i+1} - Loss: {out.item()}")

print("Ground state energy =", out.item(),"a.u")
```

In a.u, the final ground state energy is a bit higher the expected -1.851 a.u
(see page 33 of the reference paper mentioned above). Increasing the ansatz
depth is enough to reach the desired accuracy.


## References

[^1]: [Seeley et al.](https://arxiv.org/abs/1208.5986) - The Bravyi-Kitaev transformation for quantum computation of electronic structure
