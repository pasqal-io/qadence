from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import Array, grad, jit, value_and_grad, vmap
from numpy.random import uniform
from numpy.typing import ArrayLike

from qadence.backends import backend_factory
from qadence.blocks.utils import chain
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, ising_hamiltonian
from qadence.logger import get_script_logger
from qadence.types import BackendName, BasisSet, DiffMode

logger = get_script_logger("Horqrux DQC")
N_QUBITS, DEPTH, LEARNING_RATE, N_POINTS = 4, 3, 0.01, 20
logger.info(f"Running example {Path(__file__).name} with n_qubits = {N_QUBITS}")
# building the DQC model
ansatz = hea(n_qubits=N_QUBITS, depth=DEPTH)
# the input data is encoded via a feature map
fm = feature_map(n_qubits=N_QUBITS, param="x", fm_type=BasisSet.CHEBYSHEV)
# choosing a cost function
obs = ising_hamiltonian(n_qubits=N_QUBITS)
# building the circuit and the quantum model
circ = QuantumCircuit(N_QUBITS, chain(fm, ansatz))
bknd = backend_factory(BackendName.HORQRUX, DiffMode.AD)
conv_circ, conv_obs, embedding_fn, params = bknd.convert(circ, obs)

optimizer = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optimizer.init(params)


def exp_fn(params: dict[str, Array], inputs: dict[str, Array]) -> ArrayLike:
    return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))


# define a problem-specific MSE loss function
# for the ODE df/dx=4x^3+x^2-2x-1/2
def loss_fn(params: dict[str, Array], x: Array) -> Array:
    def loss(x: float) -> Array:
        dfdx = grad(lambda x: exp_fn(params, {"x": x}))(x)
        ode_loss = dfdx - (4 * x**3 + x**2 - 2 * x - 0.5)
        boundary_loss = exp_fn(params, {"x": jnp.zeros_like(x)}) - jnp.ones_like(x)
        return jnp.power(ode_loss, 2) + jnp.power(boundary_loss, 2)

    return jnp.mean(vmap(loss, in_axes=(0,))(x))


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# collocation points sampling and training
@jit
def train_step(i: int, inputs: tuple) -> tuple:
    params, opt_state = inputs
    x = jnp.array(uniform(low=-0.99, high=0.99, size=(N_POINTS, 1)))
    loss, grads = value_and_grad(loss_fn)(params, x)
    params, opt_state = optimize_step(params, opt_state, grads)
    print(f"epoch {i}, loss: {loss}")
    return params, opt_state


params, opt_state = jax.lax.fori_loop(0, 1000, train_step, (params, opt_state))
# compare the solution to known ground truth
sample_points = jnp.linspace(-1.0, 1.0, 100)
# analytical solution
analytic_sol = (
    sample_points**4
    + (1 / 3) * sample_points**3
    - sample_points**2
    - (1 / 2) * sample_points
    + 1
)
# DQC solution
dqc_sol = exp_fn(params, {"x": sample_points})
x_data = sample_points
# plot
plt.figure(figsize=(4, 4))
plt.plot(x_data, analytic_sol, color="gray", label="Exact solution")
plt.plot(x_data, dqc_sol, color="orange", label="DQC solution")
plt.xlabel("x")
plt.ylabel("df | dx")
plt.title("Simple ODE")
plt.legend()
plt.show()
