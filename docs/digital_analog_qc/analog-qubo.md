In this notebook we solve a quadratic unconstrained optimization problem with
`qadence` emulated analog interface using the QAOA variational algorithm. The
problem is detailed in the Pulser documentation
[here](https://pulser.readthedocs.io/en/stable/tutorials/qubo.html).


??? note "Construct QUBO register (defines `qubo_register_coords` function)"
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


## Define and solve QUBO

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

The QUBO is defined by weighted connections `Q` and a cost function.

```python exec="on" source="material-block" session="qubo"
def cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost


def cost_fn(counter, Q):
    cost = sum(counter[key] * cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples


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

Build a register from graph extracted from the QUBO exactly
as you would do with Pulser.
```python exec="on" source="material-block" session="qubo"
reg = Register.from_coordinates(qubo_register_coords(Q))
```

The analog circuit is composed of two global rotations per layer.  The first
rotation corresponds to the mixing Hamiltonian and the second one to the
embedding Hamiltonian.  Subsequently we add the Ising interaction term to
emulate the analog circuit.  This uses a principal quantum number n=70 for the
Rydberg level under the hood.
```python exec="on" source="material-block" result="json" session="qubo"
from qadence.transpile.emulate import ising_interaction

LAYERS = 2
block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(LAYERS)])

emulated = add_interaction(
    reg, block, interaction=lambda r, ps: ising_interaction(r, ps, rydberg_level=70)
)
print(emulated)
```

Sample the model to get the initial solution.
```python exec="on" source="material-block" session="qubo"
model = QuantumModel(QuantumCircuit(reg, emulated), backend="pyqtorch", diff_mode='gpsr')
initial_counts = model.sample({}, n_shots=1000)[0]
```

The loss function is defined by averaging over the evaluated bitstrings.
```python exec="on" source="material-block" session="qubo"
def loss(param, *args):
    Q = args[0]
    param = torch.tensor(param)
    model.reset_vparams(param)
    C = model.sample({}, n_shots=1000)[0]
    return cost_fn(C, Q)
```
Here we use a gradient-free optimization loop for reaching the optimal solution.
```python exec="on" source="material-block" result="json" session="qubo"
#
for i in range(20):
    try:
        res = minimize(
            loss,
            args=Q,
            x0=np.random.uniform(1, 10, size=2 * LAYERS),
            method="COBYLA",
            tol=1e-8,
            options={"maxiter": 20},
        )
    except Exception:
        pass

# sample the optimal solution
model.reset_vparams(res.x)
optimal_count_dict = model.sample({}, n_shots=1000)[0]
print(optimal_count_dict)
```

```python exec="on" source="material-block" html="1" session="qubo"
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# known solutions to the QUBO
solution_bitstrings=["01011", "00111"]

n_to_show = 20
xs, ys = zip(*sorted(
    initial_counts.items(),
    key=lambda item: item[1],
    reverse=True
))
colors = ["r" if x in solution_bitstrings else "g" for x in xs]

axs[0].set_xlabel("bitstrings")
axs[0].set_ylabel("counts")
axs[0].bar(xs[:n_to_show], ys[:n_to_show], width=0.5, color=colors)
axs[0].tick_params(axis="x", labelrotation=90)
axs[0].set_title("Initial solution")

xs, ys = zip(*sorted(optimal_count_dict.items(),
    key=lambda item: item[1],
    reverse=True
))
# xs = list(xs) # markdown-exec: hide
# assert (xs[0] == "01011" and xs[1] == "00111") or (xs[1] == "01011" and xs[0] == "00111"), print(f"{xs=}") # markdown-exec: hide

colors = ["r" if x in solution_bitstrings else "g" for x in xs]

axs[1].set_xlabel("bitstrings")
axs[1].set_ylabel("counts")
axs[1].bar(xs[:n_to_show], ys[:n_to_show], width=0.5, color=colors)
axs[1].tick_params(axis="x", labelrotation=90)
axs[1].set_title("Optimal solution")
plt.tight_layout()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```
