from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pulser.devices import DigitalAnalogDevice
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from qadence import (
    AnalogRX,
    AnalogRZ,
    DiffMode,
    QuantumCircuit,
    QuantumModel,
    Register,
    RydbergDevice,
    chain,
)

SHOW_PLOTS = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    print(sort_zipped[:3])

    def evaluate_mapping(new_coords, *args):
        """Cost function to minimize.

        Ideally, the pairwise distances are conserved.
        """
        Q, shape = args
        new_coords = np.reshape(new_coords, shape)
        new_Q = squareform(DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6)
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


def cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost


def cost(counter, Q):
    cost = sum(counter[key] * cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples


def plot_distribution(counter, solution_bitstrings=["01011", "00111"], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    xs, ys = zip(*sorted(counter.items(), key=lambda item: item[1], reverse=True))
    colors = ["r" if x in solution_bitstrings else "g" for x in xs]

    ax.set_xlabel("bitstrings")
    ax.set_ylabel("counts")
    ax.bar(xs, ys, width=0.5, color=colors)
    ax.tick_params(axis="x", labelrotation=90)
    return ax


fig, ax = plt.subplots(1, 2, figsize=(12, 4))

Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)


LAYERS = 2
block = chain(*[AnalogRX(f"t{i}") * AnalogRZ(f"s{i}") for i in range(LAYERS)])
device = RydbergDevice(rydberg_level=70)
reg = Register.from_coordinates(qubo_register_coords(Q), device_specs=device)
model = QuantumModel(QuantumCircuit(reg, block), diff_mode=DiffMode.GPSR)
cnts = model.sample({}, n_shots=1000)[0]

plot_distribution(cnts, ax=ax[0])


def loss(param, *args):
    Q = args[0]
    param = torch.tensor(param)
    model.reset_vparams(param)
    C = model.sample({}, n_shots=1000)[0]
    return cost(C, Q)


scores = []
params = []
for repetition in range(30):
    try:
        res = minimize(
            loss,
            args=Q,
            x0=np.random.uniform(1, 10, size=2 * LAYERS),
            method="Nelder-Mead",
            tol=1e-5,
            options={"maxiter": 20},
        )
        scores.append(res.fun)
        params.append(res.x)
    except Exception as e:
        pass

model.reset_vparams(params[np.argmin(scores)])
optimal_count_dict = model.sample({}, n_shots=1000)[0]
plot_distribution(optimal_count_dict, ax=ax[1])
plt.tight_layout()

if SHOW_PLOTS:
    plt.show()

xs, _ = zip(*sorted(optimal_count_dict.items(), key=lambda item: item[1], reverse=True))
assert (xs[0] == "01011" and xs[1] == "00111") or (xs[1] == "01011" and xs[0] == "00111"), f"{xs}"
