from __future__ import annotations

import jax.numpy as jnp
import optax
from jax import Array, jit, value_and_grad

from qadence.backends import backend_factory
from qadence.circuit import QuantumCircuit
from qadence.constructors import feature_map, hea, total_magnetization

backend = "horqrux"

num_epochs = 10
n_qubits = 4

fm = feature_map(n_qubits)
circ = QuantumCircuit(n_qubits, hea(n_qubits, depth=1))
obs = total_magnetization(n_qubits)

for diff_mode in ["ad", "gpsr"]:
    hq_bknd = backend_factory(backend, diff_mode)
    hq_circ, hq_obs, hq_embedfn, hq_init_params = hq_bknd.convert(circ, obs)
    inputs = {}
    embedded_params = hq_embedfn(hq_init_params, inputs)
    optimizer = optax.adam(learning_rate=0.001)

    param_names = embedded_params.keys()
    param_values = embedded_params.values()
    init_array = jnp.array(jnp.concatenate([arr for arr in param_values]))
    opt_state = optimizer.init(init_array)
    param_array = init_array

    def optimize_step(params: Array, opt_state: Array, grads: Array):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def _exp_fn(values: Array) -> Array:
        vals = {k: v for k, v in zip(param_names, values)}
        return hq_bknd.expectation(hq_circ, hq_obs, vals)

    init_pred = _exp_fn(init_array)

    def _loss(param_arr: Array, y_true: Array) -> Array:
        expval = _exp_fn(param_arr)
        return (expval - y_true) ** 2

    loss_and_grads = value_and_grad(_loss)

    def _train_step(param_array: Array, opt_state, y_true: Array) -> tuple:
        loss, grads = loss_and_grads(param_array, y_true)
        updated_param_array, updated_opt_state = optimize_step(param_array, opt_state, grads)
        return loss, updated_param_array, updated_opt_state

    train_step = jit(_train_step)
    y_true = jnp.array(1.0, dtype=jnp.float64)  # We need to enforce float64

    for epoch in range(num_epochs):
        loss, param_array, opt_state = train_step(param_array, opt_state, y_true)
        print(f"epoch {epoch} loss:{loss}")

    final_pred = _exp_fn(param_array)

    print(f"diff_mode '{diff_mode}, Initial prediction:")
    print(init_pred)
    print(f"diff_mode '{diff_mode}, Final prediction:")
    print(final_pred)
