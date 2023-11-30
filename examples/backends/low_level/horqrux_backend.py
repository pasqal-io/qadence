from __future__ import annotations

import jax.numpy as jnp
import optax
from jax import Array, grad, jit

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

    @jit
    def _exp_fn(values: Array) -> Array:
        vals = {k: v for k, v in zip(param_names, values)}
        return hq_bknd.expectation(hq_circ, hq_obs, vals)

    init_pred = _exp_fn(init_array)

    for _ in range(num_epochs):
        grads = grad(_exp_fn)(param_array)
        param_array, opt_state = optimize_step(param_array, opt_state, grads)

    final_pred = _exp_fn(param_array)

    print(f"diff_mode '{diff_mode}, Initial prediction:")
    print(init_pred)
    print(f"diff_mode '{diff_mode}, final prediction:")
    print(final_pred)
