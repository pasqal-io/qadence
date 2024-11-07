In this notebook, we solve a quadratic unconstrained binary optimization (QUBO) problem with
Qadence. QUBOs are very popular combinatorial optimization problems with a wide range of
applications. Here, we solve the problem using the QAOA [^1] variational algorithm by embedding
the QUBO problem weights onto a register as standard for neutral atom quantum devices.

Additional background information on QUBOs can be found
[here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html),
directly solved using the pulse-level interface Pulser.

## Define and solve QUBO

??? note "Pre-requisite: optimal register coordinates for embedding the QUBO problem"

    A basic ingredient for solving a QUBO problem with a neutral atom device is
    to embed the problem onto the atomic register. In short, embedding algorithms cast
    the problem onto a graph mapped onto the register by optimally finding atomic coordinates. A discussion on the embedding algorithms is beyond
    the scope of this tutorial and a simplified version taken from
    [here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html) is added below.

    ```python exec="on" source="material-block" session="qubo"
    import numpy as np
    import numpy.typing as npt
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from qadence import RydbergDevice

    def qubo_register_coords(Q: np.ndarray, device: RydbergDevice) -> list:
        """Compute coordinates for register."""

        def evaluate_mapping(new_coords, *args):
            """Cost function to minimize. Ideally, the pairwise
            distances are conserved"""
            Q, shape = args
            new_coords = np.reshape(new_coords, shape)
            interaction_coeff = device.rydberg_level
            new_Q = squareform(interaction_coeff / pdist(new_coords) ** 6)
            return np.linalg.norm(new_Q - Q)

        shape = (len(Q), 2)
        np.random.seed(0)
        x0 = np.random.random(shape).flatten()
        res = minimize(
            evaluate_mapping,
            x0,
            args=(Q, shape),
            method="Nelder-Mead",
            tol=1e-6,
            options={"maxiter": 200000, "maxfev": None},
        )
        return [(x, y) for (x, y) in np.reshape(res.x, (len(Q), 2))]
    ```
With the embedding routine under our belt, let's start by adding the required imports and
ensure the reproducibility of this tutorial.

```python exec="on" source="material-block" session="qubo"
import torch
from qadence import QuantumModel, QuantumCircuit, Register
from qadence import RydbergDevice, AnalogRX, AnalogRZ, chain
from qadence.ml_tools import Trainer, TrainConfig, num_parameters
import nevergrad as ng
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
```

The QUBO problem is initially defined by a graph of weighted edges and a cost function to be optimized. The weighted edges are represented by a
real-valued symmetric matrix `Q` which is used throughout the tutorial.

```python exec="on" source="material-block" session="qubo"
# QUBO problem weights (real-value symmetric matrix)
Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

def loss(model: QuantumModel, *args) -> tuple[float, dict]:
    to_arr_fn = lambda bitstring: np.array(list(bitstring), dtype=int)
    cost_fn = lambda arr: arr.T @ Q @ arr
    samples = model.sample({}, n_shots=1000)[0]  # extract samples
    cost_fn = sum(samples[key] * cost_fn(to_arr_fn(key)) for key in samples)
    return cost_fn / sum(samples.values()), {}  # We return an optional metrics dict
```

The QAOA algorithm needs a variational quantum circuit with optimizable parameters.
For that purpose, we use a fully analog circuit composed of two global rotations per layer on
different axes of the Bloch sphere.
The first rotation corresponds to the mixing Hamiltonian and the second one to the
embedding Hamiltonian [^1]. In this setting, the embedding is realized
by the appropriate register coordinates and the resulting qubit interaction.

??? note "Rydberg level"
    The Rydberg level is set to *70*. We
    initialize the weighted register graph from the QUBO definition
    similarly to what is done in the
    [original tutorial](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html),
    and set the device specifications with the updated Rydberg level.

```python exec="on" source="material-block" result="json" session="qubo"
# Device specification and atomic register
device = RydbergDevice(rydberg_level=70)

reg = Register.from_coordinates(
    qubo_register_coords(Q, device), device_specs=device
)

# Analog variational quantum circuit
layers = 2
block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(layers)])
circuit = QuantumCircuit(reg, block)
```

By feeding the circuit to a `QuantumModel` we can check the initial
counts where no clear solution can be found:

```python exec="on" source="material-block" result="json" session="qubo"
model = QuantumModel(circuit)
initial_counts = model.sample({}, n_shots=1000)[0]

print(f"initial_counts = {initial_counts}") # markdown-exec: hide
```

Finally, we can proceed with the variational optimization. The cost function
defined above is derived from bitstring computations and therefore non differentiable. We use Qadence
ML facilities to run gradient-free optimizations using the
[`nevergrad`](https://facebookresearch.github.io/nevergrad/) library.

```python exec="on" source="material-block" session="qubo"
Trainer.set_use_grad(False)

config = TrainConfig(max_iter=100)
optimizer = ng.optimizers.NGOpt(
    budget=config.max_iter, parametrization=num_parameters(model)
)
trainer = Trainer(model, optimizer, config, loss)
trainer.fit()

optimal_counts = model.sample({}, n_shots=1000)[0]
print(f"optimal_count = {optimal_counts}") # markdown-exec: hide
```

Finally, let's plot the solution. The expected bitstrings are marked in red.

```python exec="on" source="material-block" html="1" session="qubo"

# Known solutions to the QUBO problem.
solution_bitstrings = ["01011", "00111"]

def plot_distribution(C, ax, title):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = solution_bitstrings # QUBO solutions
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    ax.set_xlabel("bitstrings")
    ax.set_ylabel("counts")
    ax.set_xticks([i for i in range(len(C.keys()))], C.keys(), rotation=90)
    ax.bar(list(C.keys())[:20], list(C.values())[:20])
    ax.set_title(title)

plt.tight_layout() # markdown-exec: hide
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plot_distribution(initial_counts, axs[0], "Initial counts")
plot_distribution(optimal_counts, axs[1], "Optimal counts")
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

## References

[^1]: [Edward Farhi, Jeffrey Goldstone, Sam Gutmann, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
