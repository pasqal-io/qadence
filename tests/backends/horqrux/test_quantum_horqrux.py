from __future__ import annotations

import jax.numpy as jnp
import pytest
import torch
from jax import Array, grad, jit, value_and_grad

from qadence import (
    CNOT,
    CRX,
    CRY,
    CRZ,
    RX,
    RY,
    RZ,
    BackendName,
    QuantumCircuit,
    X,
    Y,
    Z,
    expectation,
    run,
)
from qadence.backends import backend_factory
from qadence.backends.utils import jarr_to_tensor
from qadence.blocks import AbstractBlock
from qadence.constructors import hea


def test_psr_firstOrder() -> None:
    circ = QuantumCircuit(2, hea(2, 1))
    v_list = []
    grad_dict = {}
    for diff_mode in ["ad", "gpsr"]:
        hq_bknd = backend_factory("horqrux", diff_mode)
        hq_circ, hq_obs, hq_fn, hq_params = hq_bknd.convert(circ, Z(0))
        embedded_params = hq_fn(hq_params, {})
        param_names = embedded_params.keys()
        param_values = embedded_params.values()
        param_array = jnp.array(jnp.concatenate([arr for arr in param_values]))

        def _exp_fn(values: Array) -> Array:
            vals = {k: v for k, v in zip(param_names, values)}
            return hq_bknd.expectation(hq_circ, hq_obs, vals)

        v, grads = value_and_grad(_exp_fn)(param_array)
        v_list.append(v)
        grad_dict[diff_mode] = grads
    assert jnp.allclose(grad_dict["ad"], grad_dict["gpsr"])


def test_psr_3rd_order_single_param() -> None:
    circ = QuantumCircuit(2, RX(0, "theta"))
    grad_dict = {}
    for diff_mode in ["ad", "gpsr"]:
        hq_bknd = backend_factory("horqrux", diff_mode)
        hq_circ, hq_obs, hq_fn, hq_params = hq_bknd.convert(circ, Z(0))
        embedded_params = hq_fn(hq_params, {})
        param_names = embedded_params.keys()

        def _exp_fn(value: Array) -> Array:
            vals = {list(param_names)[0]: value}
            return hq_bknd.expectation(hq_circ, hq_obs, vals)

        d1fdx = grad(_exp_fn)
        d2fdx = grad(d1fdx)
        d3fdx = grad(d2fdx)
        jd3fdx = jit(d3fdx)
        grad_dict[diff_mode] = jd3fdx(jnp.pi / 2)
    assert jnp.allclose(grad_dict["ad"], grad_dict["gpsr"])


@pytest.mark.parametrize("block", [RX(0, 1.0), RY(0, 3.0), RZ(0, 4.0), X(0), Y(0), Z(0)])
def test_singlequbit_primitive_parametric(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize(
    "block", [CNOT(0, 1), CNOT(1, 0), CRX(0, 1, 1.0), CRY(1, 0, 2.0), CRZ(0, 1, 3.0)]
)
def test_control(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize("block", [hea(2, 1), hea(4, 4)])
def test_hea(block: AbstractBlock) -> None:
    wf_pyq = run(block, backend=BackendName.PYQTORCH)
    wf_horq = jarr_to_tensor(run(block, backend=BackendName.HORQRUX))
    torch.allclose(wf_horq, wf_pyq)


@pytest.mark.parametrize("block", [hea(2, 1), hea(4, 4)])
def test_hea_expectation(block: AbstractBlock) -> None:
    exp_pyq = expectation(block, Z(0), backend=BackendName.PYQTORCH)
    exp_horq = jarr_to_tensor(
        expectation(block, Z(0), backend=BackendName.HORQRUX), dtype=torch.double
    )
    torch.allclose(exp_pyq, exp_horq)
