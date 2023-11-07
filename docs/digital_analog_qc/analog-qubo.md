In this notebook we solve a quadratic unconstrained optimization problem with
Qadence emulated analog interface using the QAOA variational algorithm. The
problem is detailed in the Pulser documentation
[here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html).

## Define and solve QUBO

??? note "Pre-requisite: construct QUBO register"
    Before we start we have to define a register that fits into our device.
    ```python exec="on" source="material-block" session="qubo"
    import torch
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, squareform

    from pulser.devices import Chadoq2

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


    def qubo_register_coords(Q):
        """Compute coordinates for register."""
        bitstrings = [np.binary_repr(i, len(Q)) for i in range(len(Q) ** 2)]
        costs = []
        # this takes exponential time with the dimension of the QUBO
        for b in bitstrings:
            z = np.array(list(b), dtype=int)
            cost = z.T @ Q @ z
            costs.append(cost)
        zipped = zip(bitstrings, costs)
        sort_zipped = sorted(zipped, key=lambda x: x[1])

        def evaluate_mapping(new_coords, *args):
            """Cost function to minimize. Ideally, the pairwise
            distances are conserved"""
            Q, shape = args
            new_coords = np.reshape(new_coords, shape)
            new_Q = squareform(Chadoq2.interaction_coeff / pdist(new_coords) ** 6)
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



```python exec="on" source="material-block" session="qubo"
import matplotlib.pyplot as plt
import numpy as np
import torch

from qadence import add_interaction, chain
from qadence import QuantumModel, QuantumCircuit, AnalogRZ, AnalogRX, Register

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
```

The QUBO problem is initially defined by a graph of weighted connections `Q` and a cost function.

```python exec="on" source="material-block" session="qubo"
def cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost

# Cost function.
def cost_fn(counter, Q):
    cost = sum(counter[key] * cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples


# Weights.
Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)
```

Now, build a weighted register graph from the QUBO definition similarly to what is
done in Pulser.

```python exec="on" source="material-block" session="qubo"
reg = Register.from_coordinates(qubo_register_coords(Q))
```

The analog circuit is composed of two global rotations per layer.  The first
rotation corresponds to the mixing Hamiltonian and the second one to the
embedding Hamiltonian in the QAOA algorithm. Subsequently, there is an Ising interaction term to
emulate the analog circuit. Please note that the Rydberg level is set to 70.

```python exec="on" source="material-block" result="json" session="qubo"
from qadence.analog.utils import ising_interaction

layers = 2
block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(layers)])

emulated = add_interaction(
    reg, block, interaction=lambda r, ps: ising_interaction(r, ps, rydberg_level=70)
)
print(f"emulated = \n") # markdown-exec: hide
print(emulated) # markdown-exec: hide
```

Next, an initial solution is computed by sampling the model:

```python exec="on" source="material-block" result="json" session="qubo"
model = QuantumModel(QuantumCircuit(reg, emulated), backend="pyqtorch", diff_mode='gpsr')
initial_counts = model.sample({}, n_shots=1000)[0]

print(f"initial_counts = {initial_counts}") # markdown-exec: hide
```

Then, the loss function is defined by averaging over the evaluated bitstrings.

```python exec="on" source="material-block" session="qubo"
def loss(param, *args):
    Q = args[0]
    param = torch.tensor(param)
    model.reset_vparams(param)
    C = model.sample({}, n_shots=1000)[0]
    return cost_fn(C, Q)
```

And a gradient-free optimization loop is used to compute the optimal solution.

```python exec="on" source="material-block" result="json" session="qubo"
# Optimization loop.
for i in range(20):
	res = minimize(
		loss,
		args=Q,
		x0=np.random.uniform(1, 10, size=2 * layers),
		method="COBYLA",
		tol=1e-8,
		options={"maxiter": 20},
	)

# Sample and visualize the optimal solution.
model.reset_vparams(res.x)
optimal_count = model.sample({}, n_shots=1000)[0]
print(f"optimal_count = {optimal_count}") # markdown-exec: hide
```

Finally, plot the solution:

```python exec="on" source="material-block" html="1" session="qubo"
fig, axs = plt.subplots(1, 2, figsize=(12, 4)) # markdown-exec: hide

# Known solutions to the QUBO problem.
solution_bitstrings=["01011", "00111"]

n_to_show = 20 # markdown-exec: hide
xs, ys = zip(*sorted(  # markdown-exec: hide
    initial_counts.items(), # markdown-exec: hide
    key=lambda item: item[1], # markdown-exec: hide
    reverse=True # markdown-exec: hide
)) # markdown-exec: hide
colors = ["r" if x in solution_bitstrings else "g" for x in xs] # markdown-exec: hide

axs[0].set_xlabel("bitstrings") # markdown-exec: hide
axs[0].set_ylabel("counts") # markdown-exec: hide
axs[0].bar(xs[:n_to_show], ys[:n_to_show], width=0.5, color=colors) # markdown-exec: hide
axs[0].tick_params(axis="x", labelrotation=90) # markdown-exec: hide
axs[0].set_title("Initial solution") # markdown-exec: hide

xs, ys = zip(*sorted(optimal_count.items(), # markdown-exec: hide
    key=lambda item: item[1], # markdown-exec: hide
    reverse=True # markdown-exec: hide
)) # markdown-exec: hide
# xs = list(xs) # markdown-exec: hide
# assert (xs[0] == "01011" and xs[1] == "00111") or (xs[1] == "01011" and xs[0] == "00111"), print(f"{xs=}") # markdown-exec: hide

colors = ["r" if x in solution_bitstrings else "g" for x in xs] # markdown-exec: hide

axs[1].set_xlabel("bitstrings") # markdown-exec: hide
axs[1].set_ylabel("counts") # markdown-exec: hide
axs[1].bar(xs[:n_to_show], ys[:n_to_show], width=0.5, color=colors) # markdown-exec: hide
axs[1].tick_params(axis="x", labelrotation=90) # markdown-exec: hide
axs[1].set_title("Optimal solution") # markdown-exec: hide
plt.tight_layout() # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```
