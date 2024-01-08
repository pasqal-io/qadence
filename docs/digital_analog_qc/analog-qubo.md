!!! warning
    Tutorial to be updated

In this notebook we solve a quadratic unconstrained binary optimization (QUBO) problem with
Qadence. QUBOs are very popular combinatorial optimization problems with a wide range of
applications. Here, we solve the problem using the QAOA [^1] variational algorithm by embedding
the QUBO problem weights onto a register. This procedure is used for solving QUBOs on
neutral atom quantum devices.

Additional background information on QUBOs can be found
[here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html)
where the same problem is solved using directly the pulse-level interface Pulser.

## Define and solve QUBO

??? note "Pre-requisite: optimal register coordinates for embedding the QUBO problem"

    A basic ingredient for solving a QUBO problem with a neutral atom device is
    to embed the problem onto the atomic register. In short, embedding algorithms cast
    the problem onto a graph and then find the appropriate atomic coordinates to load the
    problem graph onto the register. A discussion on the embedding algorithms is beyond
    the scope of this tutorial and a simplified version taken from
    [here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html) is added below.

    ```python exec="on" source="material-block" session="qubo"
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform
    from qadence import RydbergDevice

    def qubo_register_coords(Q: np.ndarray, device: RydbergDevice) -> list:
        """Compute coordinates for register."""
        bitstrings = [np.binary_repr(i, len(Q)) for i in range(len(Q) ** 2)]
        costs = []
        # this takes exponential time with the dimension of the QUBO
        for b in bitstrings:
            z = np.array(list(b), dtype=int)
            cost = z.T @ Q @ z
            costs.append(cost)

        def evaluate_mapping(new_coords: np.ndarray, *args) -> np.ndarray:
            """Cost function to minimize. Ideally, the pairwise
            distances are conserved"""
            Q, shape = args
            new_coords = np.reshape(new_coords, shape)
            rydberg_level = 70
            interaction_coeff = device.rydberg_level
            new_Q = squareform(interaction_coeff / pdist(new_coords) ** 6)
            return np.linalg.norm(new_Q - Q)

        shape = (len(Q), 2)
        costs = []
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
from qadence.ml_tools import train_gradient_free, TrainConfig, num_parameters
import nevergrad as ng

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
```

The QUBO problem is initially defined by a graph of weighted connections and a cost function to be optimized. The weighted connections are organized in a
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

# Cost function for a single bitstring
def cost_bitstring(bitstring, qubo_mat):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ qubo_mat @ z
    return cost

# Cost function for a full measured sample
def cost_qubo(counter, qubo_mat):
    cost = sum(counter[key] * cost_bitstring(key, qubo_mat) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples
```

The QAOA algorithm needs a variational quantum circuit with optimizable parameters.
We use a fully analog circuit composed of two global rotations per layer on
different axes of the Bloch sphere.
The first rotation corresponds to the mixing Hamiltonian and the second one to the
embedding Hamiltonian (given by the register coordinates and equivalent to a
free evolution of the neutral atom array) in the QAOA algorithm.

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
model = QuantumModel(circuit, backend="pyqtorch", diff_mode='gpsr')
initial_counts = model.sample({}, n_shots=1000)[0]

print(f"initial_counts = {initial_counts}") # markdown-exec: hide
```

Finally, we can proceed to the variational optimization. The cost function
defined above is based on bitstring and thus non differentiable. We use Qadence
ML facilities to run some gradient-free optimization based on the
[`nevergrad`](https://facebookresearch.github.io/nevergrad/) library.

```python exec="on" source="material-block" session="qubo"
def loss(model, *args):
    C = model.sample({}, n_shots=1000)[0]
    return cost_qubo(C, Q), {}

optimizer = ng.optimizers.NGOpt(
    budget=config.max_iter, parametrization=num_parameters(model)
)

config = TrainConfig(max_iter=100)
train_gradient_free(model, None, optimizer, config, loss)

optimal_count = model.sample({}, n_shots=1000)[0]
print(f"optimal_count = {optimal_count}") # markdown-exec: hide
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
    ax.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    ax.set_title(title)

plt.tight_layout() # markdown-exec: hide
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
plot_distribution(initial_counts, axs[0], "Initial counts")
plot_distribution(optimal_count, axs[1], "Optimal counts")
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

## References

[^1]: [Edward Farhi, Jeffrey Goldstone, Sam Gutmann, A Quantum Approximate Optimization Algorithm, arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
