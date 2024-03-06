from __future__ import annotations

import os
from typing import Callable

import jax.numpy as jnp
import optax
from jax import Array, jit, value_and_grad
from numpy.typing import ArrayLike

from qadence.backends import backend_factory
from qadence.blocks.utils import chain
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization
from qadence.logger import get_script_logger
from qadence.types import BackendName, DiffMode

logger = get_script_logger("Horqrux")
backend = BackendName.HORQRUX

num_epochs = 10
n_qubits = 4
depth = 1
logger.info(f"Running example {os.path.basename(__file__)} with n_qubits = {n_qubits}")
fm = feature_map(n_qubits)
circ = QuantumCircuit(n_qubits, chain(fm, hea(n_qubits, depth=depth)))
obs = total_magnetization(n_qubits)

for diff_mode in [DiffMode.AD, DiffMode.GPSR]:
    bknd = backend_factory(backend, diff_mode)
    conv_circ, conv_obs, embedding_fn, vparams = bknd.convert(circ, obs)
    init_params = vparams.copy()
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(vparams)

    loss: Array
    grads: dict[str, Array]  # 'grads' is the same datatype as 'params'
    inputs: dict[str, Array] = {"phi": jnp.array(1.0)}

    def optimize_step(params: dict[str, Array], opt_state: Array, grads: dict[str, Array]) -> tuple:
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def exp_fn(params: dict[str, Array], inputs: dict[str, Array] = inputs) -> ArrayLike:
        return bknd.expectation(conv_circ, conv_obs, embedding_fn(params, inputs))

    init_pred = exp_fn(vparams)

    def mse_loss(params: dict[str, Array], y_true: Array) -> Array:
        expval = exp_fn(params)
        return (expval - y_true) ** 2

    @jit
    def train_step(
        params: dict,
        opt_state: Array,
        y_true: Array = jnp.array(1.0, dtype=jnp.float64),
        loss_fn: Callable = mse_loss,
    ) -> tuple:
        loss, grads = value_and_grad(loss_fn)(params, y_true)
        params, opt_state = optimize_step(params, opt_state, grads)
        return loss, params, opt_state

    for epoch in range(num_epochs):
        loss, vparams, opt_state = train_step(vparams, opt_state)
        print(f"epoch {epoch} loss:{loss}")

    final_pred = exp_fn(vparams)

    print(
        f"diff_mode '{diff_mode}: Initial prediction: {init_pred}, initial vparams: {init_params}"
    )
    print(f"Final prediction: {final_pred}, final vparams: {vparams}")
    print("----------")
