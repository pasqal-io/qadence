# CNOT with interacting qubits

Digital-analog quantum computing focuses on using single qubit digital gates combined with more complex and device-dependent analog interactions to represent quantum programs. This paradigm has been shown to be universal for quantum computation[^1]. However, while this approach may have advantages when adapting quantum programs to real devices, known quantum algorithms are very often expressed in a fully digital paradigm. As such, it is also important to have concrete ways to transform from one paradigm to another.

This tutorial will exemplify the *DAQC transformation* starting with the representation of a simple digital CNOT using the universality of the Ising Hamiltonian[^2].

## CNOT with CPHASE

Let's look at a single example of how the digital-analog transformation can be used to perform a CNOT on two qubits inside a register of globally interacting qubits.

First, note that the CNOT can be decomposed with two Hadamard and a CPHASE gate with $\phi=\pi$:


```python exec="on" source="material-block" result="json" session="daqc-cnot"
import torch
from qadence import chain, sample, product_state

from qadence.draw import display
from qadence import X, I, Z, H, N, CPHASE, CNOT, HamEvo
from qadence.draw import html_string # markdown-exec: hide

n_qubits = 2

# CNOT gate
cnot_gate = CNOT(0, 1)

# CNOT decomposed
phi = torch.pi
cnot_decomp = chain(H(1), CPHASE(0, 1, phi), H(1))

init_state = product_state("10")

print(f"sample from CNOT gate and 100 shots = {sample(n_qubits, block=cnot_gate, state=init_state, n_shots=100)}")  # markdown-exec: hide
print(f"sample from decomposed CNOT gate and 100 shots = {sample(n_qubits, block=cnot_decomp, state=init_state, n_shots=100)}") # markdown-exec: hide
```

The CPHASE gate is fully diagonal, and can be implemented by exponentiating an Ising-like Hamiltonian, or *generator*,

$$\text{CPHASE}(i,j,\phi)=\text{exp}\left(-i\phi \mathcal{H}_\text{CP}(i, j)\right)$$

$$\begin{aligned}
\mathcal{H}_\text{CP}&=-\frac{1}{4}(I_i-Z_i)(I_j-Z_j)\\
&=-N_iN_j
\end{aligned}$$

where the number operator $N_i = \frac{1}{2}(I_i-Z_i)=\hat{n}_i$ is used, leading to an Ising-like interaction $\hat{n}_i\hat{n}_j$ that is common in neutral-atom systems. Let's rebuild the CNOT using this evolution.

```python exec="on" source="material-block" session="daqc-cnot"
from qadence import kron, block_to_tensor

# Hamiltonian for the CPHASE gate
h_cphase = (-1.0) * kron(N(0), N(1))

# Exponentiating the Hamiltonian
cphase_evo = HamEvo(h_cphase, phi)

# Check that we have the CPHASE gate:
cphase_matrix = block_to_tensor(CPHASE(0, 1, phi))
cphase_evo_matrix = block_to_tensor(cphase_evo)

assert torch.allclose(cphase_matrix, cphase_evo_matrix)
```

Now that the CPHASE generator is checked, it can be applied to the CNOT:


```python exec="on" source="material-block" result="json" session="daqc-cnot"
# CNOT with Hamiltonian Evolution
cnot_evo = chain(
    H(1),
    cphase_evo,
    H(1)
)

init_state = product_state("10")

print(f"sample cnot_gate = {sample(n_qubits, block = cnot_gate, state = init_state, n_shots = 100)}") # markdown-exec: hide
print(f"sample cnot_evo = {sample(n_qubits, block = cnot_evo, state = init_state, n_shots = 100)}") # markdown-exec: hide
```

Thus, a CNOT gate can be created by combining a few single-qubit gates together with a two-qubit Ising interaction between the control and the target qubit which is the essence of the Ising transform proposed in the seminal DAQC paper[^2] for $ZZ$ interactions. In Qadence, it works both with $ZZ$ and $NN$ interactions on equal footing.

## CNOT in an interacting system of three qubits

Consider a simple experimental setup with $n=3$ interacting qubits laid out in a triangular grid. For the sake of simplicity, all qubits interact with each other with an Ising ($NN$) interaction of constant strength $g_\text{int}$. The Hamiltonian for the system can be written by summing this interaction over all pairs:

$$\mathcal{H}_\text{sys}=\sum_{i=0}^{n}\sum_{j=0}^{i-1}g_\text{int}N_iN_j,$$

which in this case leads to only three interaction terms,

$$\mathcal{H}_\text{sys}=g_\text{int}(N_0N_1+N_1N_2+N_0N_2)$$

This generator can be easily built:


```python exec="on" source="material-block" result="json" session="daqc-cnot"
from qadence import add, kron
n_qubits = 3

# Interaction strength.
g_int = 1.0

# Build a list of interactions.
interaction_list = []
for i in range(n_qubits):
    for j in range(i):
        interaction_list.append(g_int * kron(N(i), N(j)))

h_sys = add(*interaction_list)

print(f"h_sys = {h_sys}") # markdown-exec: hide
```

Now let's consider that the experimental system is fixed, and we cannot isolate the qubits from each other. All we can do is the following:

- Turn on or off the global system Hamiltonian.
- Perform single-qubit rotations on individual qubits.

How can we perform a CNOT on two specific qubits of our choice?

To perform a *fully digital* CNOT we would need to isolate the control and target qubit from the third one and have those interact to implement the gate directly. While this may be relatively simple for a 3-qubit system, the experimental burden becomes much greater when we start going into the dozens of qubits.

However, with the digital-analog paradigm that is not the case! In fact, we can represent the two qubit Ising interaction required for the CNOT by combining the global system Hamiltonian with a specific set of single-qubit rotations. The full details of this transformation are described in the DAQC paper [^2], and it is available in `qadence` by calling the `daqc_transform` function.

The `daqc_transform` function will essentially return a program that represents the evolution of an Hamiltonian $H_\text{target}$ (*target Hamiltonian*) for a specified time $t_f$ by using only the evolution of an Hamiltonian $H_\text{build}$ (*build Hamiltonian*) for specific intervals of time together with specific single-qubit $X$ rotations. Currently, in `qadence` it is available for resource and target Hamiltonians composed only of $ZZ$ or $NN$ interactions. The generators are parsed by the `daqc_transform` function, the appropriate type is automatically determined, and the appropriate single-qubit detunings and global phases are applied.

Let's exemplify it for our CNOT problem:


```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
from qadence import daqc_transform, Strategy

# The target operation
i = 0  # Control
j = 1  # Target
k = 2  # The extra qubit

# CNOT on control and target, Identity on the extra qubit
cnot_target = kron(CNOT(i, j), I(k))

# The two-qubit Ising (NN) interaction for the CPHASE
h_int = (-1.0) * kron(N(i), N(j))

# Transforming the two-qubit Ising interaction using only our system Hamiltonian
transformed_ising = daqc_transform(
    n_qubits=3,        # Total number of qubits in the transformation
    gen_target=h_int,  # The target Ising generator
    t_f=torch.pi,      # The target evolution time
    gen_build=h_sys,   # The building block Ising generator to be used
    strategy=Strategy.sDAQC,   # Currently only sDAQC is implemented
    ignore_global_phases = False  # Global phases from mapping between Z and N
)

# display(transformed_ising)
print(html_string(transformed_ising)) # markdown-exec: hide
```

The circuit above actually only uses two evolutions of the global Hamiltonian. In the displayed circuit also see other instances of `HamEvo` which account for global-phases and single-qubit detunings related to the mapping between the $Z$ and $N$ operator. Optionally, the application of the global phases can also be ignored, as shown in the input of `daqc_transform`. This will not create exactly the same state or operator matrix in tensor form, but in practice they will be equivalent.

In general, the mapping of a $n$-qubit Ising Hamiltonian will require at most $n(n-1)$ evolutions. The transformed circuit performs these evolutions for specific times that are computed from the solution of a linear system of equations involving the set of interactions in the target and build Hamiltonians.

In this case the mapping is exact, since we used the *step-wise* DAQC technique (sDAQC). In *banged* DAQC (bDAQC) the mapping is not exact, but is easier to implement on a physical device with always-on interactions such as neutral-atom systems. Currently, only the sDAQC technique is available in `qadence`.

Just as before, we can check that using the transformed Ising circuit we exactly recover the CPHASE gate:


```python exec="on" source="material-block" session="daqc-cnot"
# CPHASE on (i, j), Identity on third qubit:
cphase_matrix = block_to_tensor(qd.kron(CPHASE(i, j, phi), I(k)))

# CPHASE using the transformed circuit:
cphase_evo_matrix = block_to_tensor(transformed_ising)

# Will fail if global phases are ignored:
assert torch.allclose(cphase_matrix, cphase_evo_matrix)
```

And we can now build the CNOT gate:

```python exec="on" source="material-block" result="json" session="daqc-cnot"
from qadence import equivalent_state

cnot_daqc = chain(
    H(j),
    transformed_ising,
    H(j)
)

# And finally run the CNOT on a specific 3-qubit initial state:
init_state = product_state("101")

# Check we get an equivalent wavefunction (will still pass if global phases are ignored)
wf_cnot = run(n_qubits, block = cnot_target, state = init_state)
wf_daqc = run(n_qubits, block = cnot_daqc, state = init_state)
assert equivalent_state(wf_cnot, wf_daqc)

# Visualize the CNOT bit-flip:
print(sample(n_qubits, block = cnot_target, state = init_state, n_shots = 100))
print(sample(n_qubits, block = cnot_daqc, state = init_state, n_shots = 100))
```

And we are done! We have effectively performed a CNOT operation on our desired target qubits by using only the global interaction of the system as the building block Hamiltonian, together with single-qubit rotations. Going through the trouble of decomposing a single digital gate into its Ising Hamiltonian is certainly not very practical, but it serves as a proof of principle for the potential of this technique to represent universal quantum computation. In the next example, we will see it applied to the digital-analog Quantum Fourier Transform.

## Technical details on the DAQC transformation

- The mapping between target generator and final circuit is performed by solving a linear system of size $n(n-1)$ where $n$ is the number of qubits, so it can be computed *efficiently* (i.e., with a polynomial cost in the number of qubits).
- The linear system to be solved is actually not invertible for $n=4$ qubits. This is very specific edge case requiring a workaround, that is currently not yet implemented.
- As mentioned, the final circuit has at most $n(n-1)$ slices, so there is at most a polynomial overhead in circuit depth.

Finally, and most important to its usage:

- The target Hamiltonian should be *sufficiently* represented in the building block Hamiltonian.

To illustrate this point, consider the following target and build Hamiltonians:

```python exec="on" source="material-block" session="daqc-cnot"
# Interaction between qubits 0 and 1
gen_target = 1.0 * (Z(0) @ Z(1))

# Fixed interaction between qubits 1 and 2, and customizable between 0 and 1
def gen_build(g_int):
    return g_int * (Z(0) @ Z(1)) + 1.0 * (Z(1) @ Z(2))
```

And now we perform the DAQC transform by setting `g_int = 1.0`, matching the target Hamiltonian:

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
transformed_ising = daqc_transform(
    n_qubits=3,
    gen_target=gen_target,
    t_f=1.0,
    gen_build=gen_build(g_int = 1.0),
)

# display(transformed_ising)
print(html_string(transformed_ising)) # markdown-exec: hide
```

And we get the transformed circuit. What if our build Hamiltonian has a very weak interaction between qubits 0 and 1?

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
transformed_ising = daqc_transform(
    n_qubits=3,
    gen_target=gen_target,
    t_f=1.0,
    gen_build=gen_build(g_int = 0.001),
)

# display(transformed_ising)
print(html_string(transformed_ising)) # markdown-exec: hide
```

As we can see, to represent the same interaction between 0 and 1, the slices using the build Hamiltonian need to evolve for much longer, since the target interaction is not sufficiently represented in the building block Hamiltonian.

In the limit where that interaction is not present at all, the transform will not work:


```python exec="on" source="material-block" result="json" session="daqc-cnot"
try:
    transformed_ising = daqc_transform(
        n_qubits=3,
        gen_target=gen_target,
        t_f=1.0,
        gen_build=gen_build(g_int = 0.0),
    )
except ValueError as error:
    print("Error:", error)
```

## References

[^1]: [Dodd et al., Universal quantum computation and simulation using any entangling Hamiltonian and local unitaries, PRA 65, 040301 (2002).](https://arxiv.org/abs/quant-ph/0106064)

[^2]: [Parra-Rodriguez et al., Digital-Analog Quantum Computation, PRA 101, 022305 (2020).](https://arxiv.org/abs/1812.03637)
