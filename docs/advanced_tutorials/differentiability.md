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

!!! warning "Only available with PyQTorch backend"
    Currently, automatic differentiation mode is only
    available when the `pyqtorch` backend is selected.

## Generalized parameter shift rule

The generalized parameter shift rule implementation in Qadence was introduced in [^3]. Here the standard parameter shift rules,
which only works for quantum operations whose generator has a single gap in its eigenvalue spectrum, was generalized
to work with arbitrary generators of quantum operations.

For this, we define the differentiable function as quantum expectation value

$$
f(x) = \left\langle 0\right|\hat{U}^{\dagger}(x)\hat{C}\hat{U}(x)\left|0\right\rangle
$$

where $\hat{U}(x)={\rm exp}{\left( -i\frac{x}{2}\hat{G}\right)}$ is the quantum evolution operator with generator $\hat{G}$ representing the structure of the underlying quantum circuit and $\hat{C}$ is the cost operator. Then using the eigenvalue spectrum $\left\{ \lambda_n\right\}$ of the generator $\hat{G}$ we calculate the full set of corresponding unique non-zero spectral gaps $\left\{ \Delta_s\right\}$ (differences between eigenvalues). It can be shown that the final expression of derivative of $f(x)$ is then given by the following expression:

$\begin{equation}
\frac{{\rm d}f\left(x\right)}{{\rm d}x}=\overset{S}{\underset{s=1}{\sum}}\Delta_{s}R_{s},
\end{equation}$

where $S$ is the number of unique non-zero spectral gaps and $R_s$ are real quantities that are solutions of a system of linear equations

$\begin{equation}
\begin{cases}
F_{1} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{1}\Delta_{s}}{2}\right)R_{s},\\
F_{2} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{2}\Delta_{s}}{2}\right)R_{s},\\
 & ...\\
F_{S} & =4\overset{S}{\underset{s=1}{\sum}}{\rm sin}\left(\frac{\delta_{M}\Delta_{s}}{2}\right)R_{s}.
\end{cases}
\end{equation}$

Here $F_s=f(x+\delta_s)-f(x-\delta_s)$ denotes the difference between values of functions evaluated at shifted arguments $x\pm\delta_s$.

## Usage

### Basics

In Qadence, the GPSR differentiation engine can be selected by passing `diff_mode="gpsr"` or, equivalently, `diff_mode=DiffMode.GPSR` to a `QuantumModel` instance. The code in the box below shows how to create `QuantumModel` instances with both AD and GPSR engines.

```python exec="on" source="material-block" session="differentiability"
from qadence import (FeatureParameter, HamEvo, X, I, Z,
                    hamiltonian_factory, QuantumCircuit,
                    QuantumModel, BackendName, DiffMode)
import torch

n_qubits = 2

# define differentiation parameter
x = FeatureParameter("x")

# define generator and HamEvo block
generator = X(0) + X(1) + 0.2 * (Z(0) + I(1)) * (I(0) + Z(1))
block = HamEvo(generator, x)

# create quantum circuit
circuit = QuantumCircuit(n_qubits, block)

# create total magnetization cost operator
obs = hamiltonian_factory(n_qubits, detuning=Z)

# create models with AD and GPSR differentiation engines
model_ad = QuantumModel(circuit, obs,
                        backend=BackendName.PYQTORCH,
                        diff_mode=DiffMode.AD)
model_gpsr = QuantumModel(circuit, obs,
                          backend=BackendName.PYQTORCH,
                          diff_mode=DiffMode.GPSR)

# generate value for circuit's parameter
xs = torch.linspace(0, 2*torch.pi, 100, requires_grad=True)
values = {"x": xs}

# calculate function f(x)
exp_val_ad = model_ad.expectation(values)
exp_val_gpsr = model_gpsr.expectation(values)

# calculate derivative df/dx using the PyTorch
# autograd engine
dexpval_x_ad = torch.autograd.grad(
    exp_val_ad, values["x"], torch.ones_like(exp_val_ad), create_graph=True
)[0]
dexpval_x_gpsr = torch.autograd.grad(
    exp_val_gpsr, values["x"], torch.ones_like(exp_val_gpsr), create_graph=True
)[0]
```

We can plot the resulting derivatives and see that in both cases they coincide.

```python exec="on" source="material-block" session="differentiability"
import matplotlib.pyplot as plt

# plot f(x) and df/dx derivatives calculated using AD and GPSR
# differentiation engines
fig, ax = plt.subplots()
ax.scatter(xs.detach().numpy(),
           exp_val_ad.detach().numpy(),
           label="f(x)")
ax.scatter(xs.detach().numpy(),
           dexpval_x_ad.detach().numpy(),
           label="df/dx AD")
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
from qadence import DifferentiableBackend
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


## References

[^1]: [A. G. Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

[^2]: [Schuld et al., Evaluating analytic gradients on quantum hardware (2018).](https://arxiv.org/abs/1811.11184)

[^3]: [Kyriienko et al., General quantum circuit differentiation rules](https://arxiv.org/abs/2108.01218)
