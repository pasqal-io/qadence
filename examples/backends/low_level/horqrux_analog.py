from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import Array, jit, value_and_grad
from numpy.typing import ArrayLike

from qadence import (
    AnalogInteraction,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    FeatureParameter,
    Register,
    VariationalParameter,
    Z,
    chain,
    hamiltonian_factory,
)
from qadence.backends import backend_factory
from qadence.circuit import QuantumCircuit
from qadence.logger import get_script_logger
from qadence.types import BackendName, DiffMode

logger = get_script_logger("horqcrux_analog")

N_QUBITS = 4
N_EPOCHS = 200
BACKEND_NAME = BackendName.HORQRUX
DIFF_MODE = DiffMode.AD

logger.info(f"Running example {os.path.basename(__file__)} with n_qubits = {N_QUBITS}")

bknd = backend_factory(BACKEND_NAME, DIFF_MODE)
register = Register.line(N_QUBITS, spacing=8.0)

# The input feature phi for the circuit to learn f(x)
phi = FeatureParameter("phi")

# Feature map with a few global analog rotations
fm = chain(
    AnalogRX(phi),
    AnalogRY(2 * phi),
    AnalogRZ(3 * phi),
)


t_0 = 1000.0 * VariationalParameter("t_0")
t_1 = 1000.0 * VariationalParameter("t_1")
t_2 = 1000.0 * VariationalParameter("t_2")

# Creating the ansatz with parameterized rotations and wait time
ansatz = chain(
    AnalogRX("tht_0"),
    AnalogRY("tht_1"),
    AnalogRZ("tht_2"),
    AnalogInteraction(t_0),
    AnalogRX("tht_3"),
    AnalogRY("tht_4"),
    AnalogRZ("tht_5"),
    AnalogInteraction(t_1),
    AnalogRX("tht_6"),
    AnalogRY("tht_7"),
    AnalogRZ("tht_8"),
    AnalogInteraction(t_2),
)

# Total magnetization observable
observable = hamiltonian_factory(N_QUBITS, detuning=Z)
# Defining the circuit and observable
circuit = QuantumCircuit(register, fm, ansatz)
conv_circ, conv_obs, embedding_fn, params = bknd.convert(circuit, observable)


# Function to fit:
def f(x: Array) -> Array:
    return x**2


x_test = jnp.linspace(-1.0, 1.0, 100)
y_test = f(x_test)
x_train = jnp.linspace(-1.0, 1.0, 10)
y_train = f(x_train)
# Initialize an optimizer
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(params)


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def exp_fn(params: dict[str, Array], inputs: dict[str, Array]) -> ArrayLike:
    return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))


# Query the model using the initial random parameter values
y_pred_initial = exp_fn(params, {"phi": x_test})


def loss_fn(params: dict[str, Array], x: Array, y: Array) -> Array:
    expval = exp_fn(params, {"phi": x})
    return jnp.mean(optax.l2_loss(jnp.ravel(expval), y))


@jit
def train_step(i: int, res: tuple) -> tuple:
    params, opt_state = res
    loss, grads = value_and_grad(loss_fn)(params, x_train, y_train)
    logger.info(f"epoch {i}: loss {loss}")
    params, opt_state = optimize_step(params, opt_state, grads)
    return params, opt_state


# Train the circuit
params, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (params, opt_state))
# Query the model using the optimal parameter values
y_pred_final = exp_fn(params, {"phi": x_test})

plt.plot(x_test, y_pred_initial, label="Initial prediction")
plt.plot(x_test, y_pred_final, label="Final prediction")
plt.scatter(x_train, y_train, label="Training points")
plt.legend()
plt.show()
