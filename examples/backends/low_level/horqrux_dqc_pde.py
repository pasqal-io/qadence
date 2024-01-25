from __future__ import annotations

from functools import reduce
from itertools import product
from operator import add

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array, jit, value_and_grad, vmap
from numpy.random import uniform
from numpy.typing import ArrayLike

from qadence.backends import backend_factory
from qadence.blocks.utils import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, ising_hamiltonian
from qadence.types import BackendName, DiffMode

LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
N_POINTS = 150
# define a simple DQC model
ansatz = hea(n_qubits=N_QUBITS, depth=DEPTH)
# parallel Fourier feature map
split = N_QUBITS // len(VARIABLES)
fm = kron(
    *[
        feature_map(n_qubits=split, support=support, param=param)
        for param, support in zip(
            VARIABLES,
            [list(list(range(N_QUBITS))[i : i + split]) for i in range(N_QUBITS) if i % split == 0],
        )
    ]
)
# choosing a cost function
obs = ising_hamiltonian(n_qubits=N_QUBITS)
# building the circuit and the quantum model
circ = QuantumCircuit(N_QUBITS, chain(fm, ansatz))
bknd = backend_factory(BackendName.HORQRUX, DiffMode.AD)
conv_circ, conv_obs, embedding_fn, params = bknd.convert(circ, obs)

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)


def exp_fn(params: dict[str, Array], inputs: dict[str, Array]) -> ArrayLike:
    return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))


def loss_fn(params: dict[str, Array], x: Array, y: Array) -> Array:
    def pde_loss(x: float, y: float) -> Array:
        l_b, r_b, t_b, b_b = list(
            map(
                lambda d: exp_fn(params, d),
                [
                    {"x": jnp.zeros(1), "y": y},  # u(0,y)=0
                    {"x": jnp.ones(1), "y": y},  # u(L,y)=0
                    {"x": x, "y": jnp.ones(1)},  # u(x,H)=0
                    {"x": x, "y": jnp.zeros(1)},  # u(x,0)=f(x)
                ],
            )
        )
        b_b -= jnp.sin(jnp.pi * x)
        hessian = jax.jacfwd(jax.grad(lambda d: exp_fn(params, d)))
        dfdxy = hessian({"x": x, "y": y})
        interior = dfdxy["x"]["x"] + dfdxy["y"]["y"]  # uxx+uyy=0
        return reduce(add, list(map(lambda t: jnp.power(t, 2), [l_b, r_b, t_b, b_b, interior])))

    return jnp.mean(vmap(pde_loss, in_axes=(0, 0))(x, y))


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# collocation points sampling and training
def sample_points(n_in: int, n_p: int) -> ArrayLike:
    return uniform(0, 1.0, (n_in, n_p))


@jit
def train_step(i: int, inputs: tuple) -> tuple:
    params, opt_state = inputs
    x, y = sample_points(2, N_POINTS)
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    params, opt_state = optimize_step(params, opt_state, grads)
    return params, opt_state


params, opt_state = jax.lax.fori_loop(0, 10, train_step, (params, opt_state))
# compare the solution to known ground truth
single_domain = jnp.linspace(0, 1, num=N_POINTS)
domain = jnp.array(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
    (np.exp(-np.pi * domain[:, 0]) * np.sin(np.pi * domain[:, 1])).reshape(N_POINTS, N_POINTS).T
)
# DQC solution

dqc_sol = vmap(lambda domain: exp_fn(params, {"x": domain[0], "y": domain[1]}), in_axes=(0,))(
    domain
).reshape(N_POINTS, N_POINTS)
# # plot results
fig, ax = plt.subplots(1, 2, figsize=(7, 7))
ax[0].imshow(analytic_sol, cmap="turbo")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Analytical solution u(x,y)")
ax[1].imshow(dqc_sol, cmap="turbo")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("DQC solution u(x,y)")
plt.show()
