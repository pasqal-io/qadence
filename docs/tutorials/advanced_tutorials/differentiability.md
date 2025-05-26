# Differentiability

Many application in quantum computing and quantum machine learning more specifically requires the differentiation
of a quantum circuit with respect to its parameters.

In Qadence, we perform quantum computations via the `QuantumModel` interface. The derivative of the outputs of quantum
models with respect to feature and variational parameters in the quantum circuit can be implemented in Qadence
with two different modes:

- Automatic differentiation (AD) mode [^1]. This mode allows to differentiation both
`run()` and `expectation()` methods of the `QuantumModel` and it is the fastest
available differentiation method. Under the hood, it is based on the PyTorch autograd engine wrapped by
the `DifferentiableBackend` class. This mode is not working on quantum devices.
- Generalized parameter shift rule (GPSR) mode. This is general implementation of the well known parameter
 shift rule algorithm [^2] which works for arbitrary quantum operations [^3]. This mode is only applicable to
 the `expectation()` method of `QuantumModel` but it is compatible with execution or quantum devices.

## Automatic differentiation

Automatic differentiation [^1] is a procedure to derive a complex function defined as a sequence of elementary
mathematical operations in
the form of a computer program. Automatic differentiation is a cornerstone of modern machine learning and a crucial
ingredient of its recent successes. In its so-called *reverse mode*, it follows this sequence of operations in reverse order by systematically applying the chain rule to recover the exact value of derivative. Reverse mode automatic differentiation
is implemented in Qadence leveraging the PyTorch `autograd` engine.

!!! warning "Only available via the PyQTorch or Horqrux backends"
    Currently, automatic differentiation mode is only
    available when the `pyqtorch` or `horqrux` backends are selected.

## Generalized parameter shift rule

The generalized parameter shift rule implementation in Qadence was introduced in [^3]. Here the standard parameter shift rules,
which only works for quantum operations whose generator has a single gap in its eigenvalue spectrum, was generalized
to work with arbitrary generators of quantum operations.

For this, we define the differentiable function as quantum expectation value

$$
f(x) = \left\langle 0\right|\hat{U}^{\dagger}(x)\hat{C}\hat{U}(x)\left|0\right\rangle
$$

where $\hat{U}(x)={\rm exp}{\left( -i\frac{x}{2}\hat{G}\right)}$ is the quantum evolution operator
with generator $\hat{G}$ representing the structure of the underlying quantum circuit and $\hat{C}$ is the cost operator.
Then using the eigenvalue spectrum $\lambda_n$ of the generator $\hat{G}$
we calculate the full set of corresponding unique non-zero spectral gaps ${ \Delta_s\}$ (differences between eigenvalues).
It can be shown that the final expression of derivative of $f(x)$ is then given by the following expression:

$$
\begin{equation}
\frac{{\rm d}f\left(x\right)}{{\rm d}x}=\overset{S}{\underset{s=1}{\sum}}\Delta_{s}R_{s},
\end{equation}
$$

where $S$ is the number of unique non-zero spectral gaps and $R_s$ are real quantities that are solutions of a system of linear equations

$$
\begin{equation}
\begin{cases}
F_{1} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{1}\Delta_{s}}{2}\right)R_{s},\\
F_{2} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{2}\Delta_{s}}{2}\right)R_{s},\\
 & ...\\
F_{S} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{M}\Delta_{s}}{2}\right)R_{s}.
\end{cases}
\end{equation}
$$

Here $F_s=f(x+\delta_s)-f(x-\delta_s)$ denotes the difference between values of functions evaluated at shifted arguments $x\pm\delta_s$.

### Approximate Generalized parameter shift rule

The approximate generalized parameter shift rule (aGPSR) implementation in Qadence was introduced in [^4]. The aGPSR has been proposed as method
of estimating derivative of a function spawned by an arbitrary generator having a non-trivial spectrum
of eigenvalues in a limited shot budget setting. The idea is to reduce significantly the number of gaps involved in the system of equations above.
Hence, we introduce using pseudo-gaps $\{\delta_k\}_{k=1}^K$, with $K << S$. aGPSR is very interesting when using analog operations as we can reduce significantly the number of expectation calls.

## Adjoint Differentiation
Qadence also offers a memory-efficient, non-device compatible alternative to automatic differentation, called 'Adjoint Differentiation' [^5] and allows for precisely calculating the gradients of variational parameters in O(P) time and using O(1) state-vectors. Adjoint Differentation is currently only supported by the Torch Engine and allows for first-order derivatives only.

## Usage

### Basics

In Qadence, the differentiation modes can be selected via the `diff_mode` argument of the QuantumModel class. It either accepts a `DiffMode`(`DiffMode.GSPR`, `DiffMode.AD` or `DiffMode.ADJOINT`) or a string (`"gpsr""`, `"ad"` or `"adjoint"`). The code in the box below shows how to create `QuantumModel` instances with all available differentiation modes.

```python exec="on" source="material-block" session="differentiability"
from qadence import (FeatureParameter, RX, Z, hea, chain,
                    hamiltonian_factory, QuantumCircuit,
                    QuantumModel, BackendName, DiffMode)
import torch

n_qubits = 2

# Define a symbolic parameter to differentiate with respect to
x = FeatureParameter("x")

block = chain(hea(n_qubits, 1), RX(0, x))

# create quantum circuit
circuit = QuantumCircuit(n_qubits, block)

# create total magnetization cost operator
obs = hamiltonian_factory(n_qubits, detuning=Z)

# create models with AD, ADJOINT and GPSR differentiation engines
model_ad = QuantumModel(circuit, obs,
                        backend=BackendName.PYQTORCH,
                        diff_mode=DiffMode.AD)
model_adjoint = QuantumModel(circuit, obs,
                        backend=BackendName.PYQTORCH,
                        diff_mode=DiffMode.ADJOINT)
model_gpsr = QuantumModel(circuit, obs,
                          backend=BackendName.PYQTORCH,
                          diff_mode=DiffMode.GPSR)

# Create concrete values for the parameter we want to differentiate with respect to
xs = torch.linspace(0, 2*torch.pi, 100, requires_grad=True)
values = {"x": xs}

# calculate function f(x)
exp_val_ad = model_ad.expectation(values)
exp_val_adjoint = model_adjoint.expectation(values)
exp_val_gpsr = model_gpsr.expectation(values)

# calculate derivative df/dx using the PyTorch
# autograd engine
dexpval_x_ad = torch.autograd.grad(
    exp_val_ad, values["x"], torch.ones_like(exp_val_ad), create_graph=True
)[0]
dexpval_x_adjoint = torch.autograd.grad(
    exp_val_adjoint, values["x"], torch.ones_like(exp_val_ad), create_graph=True
)[0]
dexpval_x_gpsr = torch.autograd.grad(
    exp_val_gpsr, values["x"], torch.ones_like(exp_val_gpsr), create_graph=True
)[0]
```

We can plot the resulting derivatives and see that in both cases they coincide.

```python exec="on" source="material-block" session="differentiability"
import matplotlib.pyplot as plt

# plot f(x) and df/dx derivatives calculated using AD ,ADJOINT and GPSR
# differentiation engines
fig, ax = plt.subplots()
ax.scatter(xs.detach().numpy(),
           exp_val_ad.detach().numpy(),
           label="f(x)")
ax.scatter(xs.detach().numpy(),
           dexpval_x_ad.detach().numpy(),
           label="df/dx AD")
ax.scatter(xs.detach().numpy(),
           dexpval_x_adjoint.detach().numpy(),
           label="df/dx ADJOINT")
ax.scatter(xs.detach().numpy(),
           dexpval_x_gpsr.detach().numpy(),
           s=5,
           label="df/dx GPSR")
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


### Low-level control on the shift values

In order to get a finer control over the GPSR differentiation engine we can use the low-level Qadence API to define a `DifferentiableBackend`.

```python exec="on" source="material-block" session="differentiability"
from qadence.engines.torch import DifferentiableBackend
from qadence.backends.pyqtorch import Backend as PyQBackend

# define differentiable quantum backend
quantum_backend = PyQBackend()
conv = quantum_backend.convert(circuit, obs)
pyq_circ, pyq_obs, embedding_fn, params = conv
diff_backend = DifferentiableBackend(quantum_backend, diff_mode=DiffMode.GPSR, shift_prefac=0.2)

# calculate function f(x)
expval = diff_backend.expectation(pyq_circ, pyq_obs, embedding_fn(params, values))
```

Here we passed an additional argument `shift_prefac` to the `DifferentiableBackend` instance that governs the magnitude of shifts $\delta\equiv\alpha\delta^\prime$ shown in equation (2) above. In this relation $\delta^\prime$ is set internally and $\alpha$ is the value passed by `shift_prefac` and the resulting shift value $\delta$ is then used in all the following GPSR calculations.

Tuning parameter $\alpha$ is useful to improve results
when the generator $\hat{G}$ or the quantum operation is a dense matrix, for example a complex `HamEvo` operation; if many entries of this matrix are sufficiently larger than 0 the operation is equivalent to a strongly interacting system. In such case parameter $\alpha$ should be gradually lowered in order to achieve exact derivative values.

### Using the approximate Generalized parameter shift rule

To use aGPSR, we can simply specify the number of pseudo-gaps using a dictionary with a key `n_eqs` when configuring a model.
For the model, we use the same configuration used in the aGPSR paper [^4].

```python exec="on" source="material-block" session="differentiability"
from qadence import HamEvo, add, Register, Parameter, X, N, Y
from qadence.analog.constants import C6_DICT
from math import cos, sin

config = {
    "n_eqs": 4,
    "gap_step": 3.0,
}

def create_analog_circuit(n_qubits: int):
    """Create a circuit with one analog operation similar to the aGPSR paper"""
    spacing = 7.0
    omega = 5
    detuning = 0
    phase = 0.0

    # differentiable param
    x = Parameter("x", trainable=False)

    # define register
    register = Register.rectangular_lattice(n_qubits, 1, spacing=spacing)


    # Building the terms in the driving Hamiltonian
    h_x = add((omega * (i*0.+1) / 2) * cos(phase) * X(i) for i in range(n_qubits))
    h_y = add((-1.0 * omega * (i*0.+1) / 2) * sin(phase) * Y(i) for i in range(n_qubits))
    h_n = -1.0 * detuning * add(N(i) for i in range(n_qubits))

    # Building the interaction Hamiltonian

    # Dictionary of coefficient values for each Rydberg level, which is 60 by default
    c_6 = C6_DICT[60]
    h_int = c_6 * (
        1/(spacing**6) * (N(1)@N(0))
    )
    for i in range(2, n_qubits):
        for j in range(i):
            s = (i - j) * spacing
            h_int += c_6 * (
                1/(s**6) * (N(i)@N(j))
            )

    hamiltonian = h_x + h_y + h_n + h_int


    # Convert duration to µs due to the units of the Hamiltonian
    block = HamEvo(hamiltonian, x / omega)

    circ = QuantumCircuit(register, block)
    return circ

model_agpsr = QuantumModel(create_analog_circuit(n_qubits), obs,
                          backend=BackendName.PYQTORCH,
                          diff_mode=DiffMode.GPSR, configuration=config)

exp_val_agpsr = model_gpsr.expectation(values)
dexpval_x_gpsr = torch.autograd.grad(
    exp_val_agpsr, values["x"], torch.ones_like(exp_val_agpsr), create_graph=True
)[0]
```

### Low-level differentiation of qadence circuits using JAX
For users interested in using the `JAX` engine instead, we show how to run and differentiate qadence programs using the `horqrux` backend under [qadence examples](https://github.com/pasqal-io/qadence/tree/main/examples/backends/low_level).



## Parametrized observable differentiation

To allow differentiating observable parameters only, we need to specify the `values` argument as a dictionary with one of the two keys `circuit` and `observables`, each being a dictionary of corresponding parameters and values:

```python exec="on" source="material-block" session="differentiability"
parametric_obs = "z" * obs
z = torch.tensor([2.0], requires_grad=True)
values = {"circuit": {"x": xs}, "observables": {"z": z}}

model_ad = QuantumModel(
    circuit, parametric_obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
)
exp_val_ad = model_ad.expectation(values)

dexpval_z_ad = torch.autograd.grad(
    exp_val_ad, z, torch.ones_like(exp_val_ad), create_graph=True
)[0]
```

!!! warning "Only available via the PyQTorch backend"
    Currently, differentiating with separated parameters is only
    possible when the `pyqtorch` backend is selected.

### Differentiating only with circuit parameters

We can also specify only the `circuit` key if the observable has no parameters.

```python exec="on" source="material-block" session="differentiability"
obs = hamiltonian_factory(n_qubits, detuning=Z)
xs = torch.linspace(0, 2*torch.pi, 100, requires_grad=True)
values = {"circuit": {"x": xs}}

model_ad = QuantumModel(
    circuit, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
)
exp_val_ad = model_ad.expectation(values)

dexpval_x_ad = torch.autograd.grad(
    exp_val_ad, values["circuit"]["x"], torch.ones_like(exp_val_ad), create_graph=True
)[0]
```

### Differentiating only with observable parameters

We can also specify only the `observables` key if the circuit has no parameters.


```python exec="on" source="material-block" session="differentiability"
block = chain(
    hea(n_qubits, 1), RX(0, torch.rand(1, requires_grad=False))
)
circuit = QuantumCircuit(n_qubits, block)

parametric_obs = "z" * obs
z = torch.tensor([2.0], requires_grad=True)
values = {"observables": {"z": z}}

model_ad = QuantumModel(
    circuit, parametric_obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
)
exp_val_ad = model_ad.expectation(values)

dexpval_z_ad = torch.autograd.grad(
    exp_val_ad,
    values["observables"]["z"],
    torch.ones_like(exp_val_ad),
    create_graph=True,
)[0]
```



## References

[^1]: [A. G. Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

[^2]: [Schuld et al., Evaluating analytic gradients on quantum hardware (2018).](https://arxiv.org/abs/1811.11184)

[^3]: [Kyriienko et al., General quantum circuit differentiation rules](https://arxiv.org/abs/2108.01218)

[^4]: [Abramavicius et al., Evaluation of derivatives using approximate generalized parameter shift rule](https://arxiv.org/abs/2505.18090)

[^5]: [Tyson et al., Efficient calculation of gradients in classical simulations of variational quantum algorithms](https://arxiv.org/abs/2009.02823)
