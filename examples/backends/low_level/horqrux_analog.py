from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import Array, grad, jit
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
from qadence.types import BackendName

backend = BackendName.HORQRUX
# Line register
n_qubits = 2
register = Register.line(n_qubits, spacing=8.0)

# The input feature x for the circuit to learn f(x)
x = FeatureParameter("x")

# Feature map with a few global analog rotations
fm = chain(
    AnalogRX(x),
    AnalogRY(2 * x),
    AnalogRZ(3 * x),
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
observable = hamiltonian_factory(n_qubits, detuning=Z)

# Defining the circuit and observable
circuit = QuantumCircuit(register, fm, ansatz)

bknd = backend_factory(backend, "ad")
conv_circ, conv_obs, embedding_fn, vparams = bknd.convert(circuit, observable)
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(vparams)

loss: Array
grads: dict[str, Array]  # 'grads' is the same datatype as 'params'
inputs: dict[str, Array]


# Function to fit:
def f(x: Array) -> Array:
    return x**2


x_test = jnp.linspace(-1.0, 1.0, 100)
y_test = f(x_test)

x_train = jnp.linspace(-1.0, 1.0, 10)
y_train = f(x_train)


def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def exp_fn(params: dict[str, Array], inputs: dict[str, Array]) -> ArrayLike:
    return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))


inputs = {"x": x_train}
y_pred_initial = exp_fn(vparams, {"x": x_test})


def mse_loss(params: dict[str, Array], y_true: Array = y_train) -> Array:
    expval = exp_fn(params, inputs)
    return jnp.mean((expval - y_true) ** 2)


@jit
def train_step(i: int, res: tuple) -> tuple:
    vparams, opt_state = res
    grads = grad(mse_loss)(vparams)
    vparams, opt_state = optimize_step(vparams, opt_state, grads)
    return vparams, opt_state


N_EPOCHS = 200
vparams, opt_state = jax.lax.fori_loop(0, N_EPOCHS, train_step, (vparams, opt_state))
y_pred_final = exp_fn(vparams, {"x": x_test})

plt.plot(x_test, y_pred_initial, label="Initial prediction")
plt.plot(x_test, y_pred_final, label="Final prediction")
plt.scatter(x_train, y_train, label="Training points")
plt.legend()
plt.show()
