from __future__ import annotations

from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array, jit, vmap
from numpy.random import uniform
from numpy.typing import ArrayLike

from qadence.backends import backend_factory
from qadence.blocks.utils import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, ising_hamiltonian
from qadence.logger import get_script_logger
from qadence.types import BackendName, DiffMode

logger = get_script_logger("Horqrux_dqc_pde")

LEARNING_RATE = 0.01
N_QUBITS = 4
DEPTH = 3
VARIABLES = ("x", "y")
BATCH_SIZE = 150
NUM_VARIABLES = len(VARIABLES)
X_POS = 0
Y_POS = 1
logger.info(f"Running example {Path(__file__).name} with n_qubits = {N_QUBITS}")
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
bknd = backend_factory(BackendName.HORQRUX, DiffMode.GPSR)
conv_circ, conv_obs, embedding_fn, params = bknd.convert(circ, obs)

optimizer = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optimizer.init(params)


def exp_fn(params: dict[str, Array], x, y) -> ArrayLike:
    return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, {"x": x, "y": y}))


def loss_fn(params: dict) -> Array:
    def pde_loss(x: Array, y: Array) -> Array:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        left = (jnp.zeros_like(y), y)  # u(0,y)=0
        right = (jnp.ones_like(y), y)  # u(L,y)=0
        top = (x, jnp.ones_like(x))  # u(x,H)=0
        bottom = (x, jnp.zeros_like(x))  # u(x,0)=f(x)
        terms = jnp.dstack(list(map(jnp.hstack, [left, right, top, bottom])))
        loss_left, loss_right, loss_top, loss_bottom = vmap(
            lambda xy: exp_fn(params, xy[:, 0], xy[:, 1]), in_axes=(2,)
        )(terms)
        loss_bottom -= jnp.sin(jnp.pi * x)
        hessian = jax.hessian(lambda xy: exp_fn(params, xy[0], xy[1]))(
            jnp.concatenate(
                [
                    x.reshape(
                        1,
                    ),
                    y.reshape(
                        1,
                    ),
                ]
            )
        )
        loss_interior = hessian[X_POS][X_POS] + hessian[Y_POS][Y_POS]  # uxx+uyy=0
        return jnp.sum(
            jnp.concatenate(
                list(
                    map(
                        lambda term: jnp.power(term, 2).reshape(-1, 1),
                        [loss_left, loss_right, loss_top, loss_bottom, loss_interior],
                    )
                )
            )
        )

    return jnp.mean(
        vmap(pde_loss, in_axes=(0, 0))(*np.random.uniform(0, 1.0, (NUM_VARIABLES, BATCH_SIZE)))
    )


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
    grads = jax.grad(loss_fn)(params)
    params, opt_state = optimize_step(params, opt_state, grads)
    return params, opt_state


params, opt_state = jax.lax.fori_loop(0, 10, train_step, (params, opt_state))
# compare the solution to known ground truth
single_domain = jnp.linspace(0, 1, num=BATCH_SIZE)
domain = jnp.array(list(product(single_domain, single_domain)))
# analytical solution
analytic_sol = (
    (np.exp(-np.pi * domain[:, 0]) * np.sin(np.pi * domain[:, 1])).reshape(BATCH_SIZE, BATCH_SIZE).T
)
# DQC solution

dqc_sol = vmap(lambda domain: exp_fn(params, domain[0], domain[1]), in_axes=(0,))(domain).reshape(
    BATCH_SIZE, BATCH_SIZE
)
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
# plt.show()
