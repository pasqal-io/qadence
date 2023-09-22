# Digital-Analog QFT (Advanced)

Following the work in the DAQC paper [^1], the authors also proposed an algorithm using this technique to perform the well-known Quantum Fourier Transform [^2]. In this tutorial we will go over how the Ising transform used in the DAQC technique can be used to recreate the results for the DA-QFT.

## The (standard) digital QFT

The standard Quantum Fourier Transform can be easily built in `qadence` by calling the `qft` function. It accepts three arguments:

- `reverse_in` (default `False`): reverses the order of the input qubits
- `swaps_out` (default `False`): swaps the qubit states at the output
- `inverse` (default `False`): performs the inverse QFT


```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
import torch
import qadence as qd

from qadence.draw import display
from qadence import X, I, Z, H, CPHASE, CNOT, HamEvo
from qadence.draw import html_string # markdown-exec: hide

n_qubits = 4

qft_circuit = qd.qft(n_qubits)

display(qft_circuit)
print(html_string(qft_circuit)) # markdown-exec: hide
```

Most importantly, the circuit has a layered structure. The QFT for $n$ qubits has a total of $n$ layers, and each layer starts with a Hadamard gate on the first qubit and then builds a ladder of `CPHASE` gates. Let's see how we can easily build a function to replicate this circuit.

```python exec="on" source="material-block" session="daqc-cnot"
def qft_layer(n_qubits, layer_ix):
    qubit_range = range(layer_ix + 1, n_qubits)
    # CPHASE ladder
    cphases = []
    for j in qubit_range:
        angle = torch.pi / (2 ** (j - layer_ix))
        cphases.append(CPHASE(j, layer_ix, angle))
    # Return Hadamard followed by CPHASEs
    return qd.chain(H(layer_ix), *cphases)
```

With the layer function we can easily write the full QFT:

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
def qft_digital(n_qubits):
    return qd.chain(qft_layer(n_qubits, i) for i in range(n_qubits))

qft_circuit = qft_digital(4)

display(qft_circuit)
print(html_string(qft_circuit)) # markdown-exec: hide
```

## Decomposing the CPHASE ladder

As we already saw in the [previous DAQC tutorial](daqc-cnot.md), the CPHASE gate has a well-known decomposition into an Ising Hamiltonian. For the CNOT example, we used the decomposition into $NN$ interactions. However, here we will use the decomposition into $ZZ$ interactions to be consistent with the description in the original DA-QFT paper [^2]. The decomposition is the following:

$$\text{CPHASE}(i,j,\phi)=\text{exp}\left(-i\phi H_\text{CP}(i, j)\right)$$

$$\begin{aligned}
H_\text{CP}&=-\frac{1}{4}(I_i-Z_i)(I_j-Z_j)\\
&=-\frac{1}{4}(I_iI_j-Z_i-Z_j)-\frac{1}{4}Z_iZ_j
\end{aligned}$$

where the terms in $(I_iI_j-Z_i-Z_j)$ represents single-qubit rotations, while the interaction is given by the Ising term $Z_iZ_j$.

Just as we did for the CNOT, to build the DA-QFT we need to write the CPHASE ladder as an Ising Hamiltonian. To do so, we again write the Hamiltonian consisting of the single-qubit rotations from all CPHASEs in the layer, as well as the Hamiltonian for the two-qubit Ising interactions so that we can then use the DAQC transformation. The full mathematical details for this are written in the paper [^2], and below we write the necessary code for it, using the same notation as in the paper, including indices running from 1 to N.


```python exec="on" source="material-block" session="daqc-cnot"
# The angle of the CPHASE used in the single-qubit rotations:
def theta(k):
    """Eq. (16) from [^2]."""
    return torch.pi / (2 ** (k + 1))

# The angle of the CPHASE used in the two-qubit interactions:
def alpha(c, k, m):
    """Eq. (16) from [^2]."""
    return torch.pi / (2 ** (k - m + 2)) if c == m else 0.0
```

The first two functions represent the angles of the various `CPHASE` gates that will be used to build the qubit Hamiltonians representing each QFT layer. In the `alpha` function we include an implicit kronecker delta between the indices `m` and `c`, following the conventions and equations written in the paper [^2]. This is simply because when building the Hamiltonian the paper sums through all possible $n(n-1)$ interacting pairs, but only the pairs that are connected by a `CPHASE` in each QFT layer should have a non-zero interaction.


```python exec="on" source="material-block" session="daqc-cnot"
# Building the generator for the single-qubit rotations
def build_sqg_gen(n_qubits, m):
    """Generator in Eq. (13) from [^2] without the Hadamard."""
    k_sqg_range = range(2, n_qubits - m + 2)
    sqg_gen_list = []
    for k in k_sqg_range:
        sqg_gen = qd.kron(I(j) for j in range(n_qubits)) - Z(k+m-2) - Z(m-1)
        sqg_gen_list.append(theta(k) * sqg_gen)
    return sqg_gen_list

# Building the generator for the two-qubit interactions
def build_tqg_gen(n_qubits, m):
    """Generator in Eq. (14) from [^2]."""
    k_tqg_range = range(2, n_qubits + 1)
    tqg_gen_list = []
    for k in k_tqg_range:
        for c in range(1, k):
            tqg_gen = qd.kron(Z(c-1), Z(k-1))
            tqg_gen_list.append(alpha(c, k, m) * tqg_gen)
    return tqg_gen_list
```

There's a lot to process in the above functions, and it might be worth taking some time to go through them with the help of the description in [^2].

Let's convince ourselves that they are doing what they are supposed to: perform one layer of the QFT using a decomposition of the CPHASE gates into an Ising Hamiltonian. We start by defining the function that will produce a given QFT layer:


```python exec="on" source="material-block" session="daqc-cnot"
def qft_layer_decomposed(n_qubits, layer_ix):
    m  = layer_ix + 1 # Paper index convention

    # Step 1:
    # List of generator terms for the single-qubit rotations
    sqg_gen_list = build_sqg_gen(n_qubits, m)
    # Exponentiate the generator for single-qubit rotations:
    sq_rotations = HamEvo(qd.add(*sqg_gen_list), -1.0)

    # Step 2:
    # List of generator for the two-qubit interactions
    ising_gen_list = build_tqg_gen(n_qubits, m)
    # Exponentiating the Ising interactions:
    ising_cphase = HamEvo(qd.add(*ising_gen_list), -1.0)

    # Add the explicit Hadamard to start followed by the Hamiltonian evolutions
    if len(sqg_gen_list) > 0:
        return qd.chain(H(layer_ix), sq_rotations, ising_cphase)
    else:
        # If the generator lists are empty returns just the Hadamard of the final layer
        return H(layer_ix)
```

And now we build a layer of the QFT for both the digital and the decomposed case and check that they match:

```python exec="on" source="material-block" session="daqc-cnot"
n_qubits = 3
layer_ix = 0

# Building the layer with the digital QFT:
digital_layer_block = qft_layer(n_qubits, layer_ix)

# Building the layer with the Ising decomposition:
decomposed_layer_block = qft_layer_decomposed(n_qubits, layer_ix)

# Check that we get the same block in matrix form:
block_digital_matrix = qd.block_to_tensor(digital_layer_block)
block_decomposed_matrix = qd.block_to_tensor(decomposed_layer_block)

assert torch.allclose(block_digital_matrix, block_decomposed_matrix)
```

## Performing the DAQC transformation

We now have all the ingredients to build the Digital-Analog QFT:

- In the [previous DAQC tutorial](daqc-cnot.md) we have learned about transforming an arbitrary Ising Hamiltonian into a program executing only a fixed, system-specific one.
- In this tutorial we have so far learned how to "extract" the arbitrary Ising Hamiltonian being used in each QFT layer.

All that is left for us to do is to specify our system Hamiltonian, apply the DAQC transform, and build the Digital-Analog QFT layer function.

For simplicity, we will once again consider an all-to-all Ising Hamiltonian with a constant interaction strength, but this step generalizes so any other Hamiltonian (given the limitations already discussed in the [previous DAQC tutorial](daqc-cnot.md)).

```python exec="on" source="material-block" session="daqc-cnot"
def h_sys(n_qubits, g_int = 1.0):
    interaction_list = []
    for i in range(n_qubits):
        for j in range(i):
            interaction_list.append(g_int * qd.kron(Z(i), Z(j)))
    return qd.add(*interaction_list)
```

Now, all we have to do is re-write the qft layer function but replace Step 2. with the transformed evolution:

```python exec="on" source="material-block" session="daqc-cnot"
def qft_layer_DAQC(n_qubits, layer_ix):
    m  = layer_ix + 1 # Paper index convention

    # Step 1:
    # List of generator terms for the single-qubit rotations
    sqg_gen_list = build_sqg_gen(n_qubits, m)
    # Exponentiate the generator for single-qubit rotations:
    sq_rotations = HamEvo(qd.add(*sqg_gen_list), -1.0)

    # Step 2:
    # List of generator for the two-qubit interactions
    ising_gen_list = build_tqg_gen(n_qubits, m)
    # Transforming the target generator with DAQC:
    gen_target = qd.add(*ising_gen_list)

    transformed_ising = qd.daqc_transform(
        n_qubits = n_qubits,          # Total number of qubits in the transformation
        gen_target = gen_target,      # The target Ising generator
        t_f = -1.0,                   # The target evolution time
        gen_build = h_sys(n_qubits),  # The building block Ising generator to be used
    )

    # Add the explicit Hadamard to start followed by the Hamiltonian evolutions
    if len(sqg_gen_list) > 0:
        return qd.chain(H(layer_ix), sq_rotations, transformed_ising)
    else:
        # If the generator lists are empty returns just the Hadamard of the final layer
        return H(layer_ix)
```

And finally, to convince ourselves that the results are correct, let's build the full DA-QFT and compare it with the digital version:

```python exec="on" source="material-block" html="1" session="daqc-cnot"
def qft_digital_analog(n_qubits):
    return qd.chain(qft_layer_DAQC(n_qubits, i) for i in range(n_qubits))

n_qubits = 3

digital_qft_block = qft_digital(n_qubits)

daqc_qft_block = qft_digital_analog(n_qubits)

# Check that we get the same block in matrix form:
block_digital_matrix = qd.block_to_tensor(digital_qft_block)
block_daqc_matrix = qd.block_to_tensor(daqc_qft_block)

assert torch.allclose(block_digital_matrix, block_daqc_matrix)
```

And we can now display the program for the DA-QFT:

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"

display(daqc_qft_block)
print(html_string(daqc_qft_block)) # markdown-exec: hide
```

## The DA-QFT in `qadence`:

The digital-analog QFT is available directly by using the `strategy` argument in the QFT:

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
n_qubits = 3

qft_circuit = qd.qft(n_qubits, strategy = qd.Strategy.SDAQC)

display(qft_circuit)
print(html_string(qft_circuit)) # markdown-exec: hide
```

Just like with the `daqc_transform`, we can pass a different build Hamiltonian to it for the analog blocks, including one composed of $NN$ interactions:

```python exec="on" source="material-block" html="1" result="json" session="daqc-cnot"
from qadence import nn_hamiltonian

n_qubits = 3

gen_build = nn_hamiltonian(n_qubits)

qft_circuit = qd.qft(n_qubits, strategy = qd.Strategy.SDAQC, gen_build = gen_build)

display(qft_circuit)
print(html_string(qft_circuit)) # markdown-exec: hide
```

## References

[^1]: [Parra-Rodriguez et al., Digital-Analog Quantum Computation. PRA 101, 022305 (2020).](https://arxiv.org/abs/1812.03637)

[^2]: [Martin, Ana, et al. Digital-analog quantum algorithm for the quantum Fourier transform. Phys. Rev. Research 2.1, 013012 (2020).](https://arxiv.org/abs/1906.07635)
