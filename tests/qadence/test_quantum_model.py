from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import strategies as st  # type: ignore
import torch
from hypothesis import given, settings
from metrics import ADJOINT_ACCEPTANCE, ATOL_DICT, JS_ACCEPTANCE  # type: ignore

from qadence.backends.api import backend_factory
from qadence.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from qadence.blocks import AbstractBlock, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import hea, total_magnetization
from qadence.divergences import js_divergence
from qadence.ml_tools.utils import rand_featureparameters
from qadence.model import QuantumModel
from qadence.operations import MCRX, RX, HamEvo, I, Toffoli, X, Z
from qadence.parameters import FeatureParameter, VariationalParameter
from qadence.states import equivalent_state
from qadence.transpile import invert_endianness
from qadence.types import PI, BackendName, DiffMode

np.random.seed(42)
torch.manual_seed(42)


def digital_analog_circ(n_qubits: int = 2, depth: int = 1) -> QuantumCircuit:
    t_evo = VariationalParameter("tevo")
    g_evo = FeatureParameter("gevo")

    feature_map = HamEvo(g_evo, t_evo, qubit_support=tuple(range(n_qubits)))
    ansatz = hea(n_qubits=n_qubits, depth=depth)

    return QuantumCircuit(n_qubits, feature_map, ansatz)


def test_quantum_model_parameters(parametric_circuit: QuantumCircuit) -> None:
    circ = parametric_circuit
    assert len(circ.unique_parameters) == 4
    model_psr = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    model_ad = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    assert len([i for i in model_psr.parameters()]) == 4
    assert len([i for i in model_ad.parameters()]) == 4
    embedded_params_psr = model_psr.embedding_fn(model_psr._params, {"x": torch.rand(1)})
    embedded_params_ad = model_ad.embedding_fn(model_ad._params, {"x": torch.rand(1)})
    assert (
        len(embedded_params_ad) == 5 + 1
    )  # adding one because original param x is included for PYQ + AD
    assert len(embedded_params_psr) == 6


def test_quantum_model_duplicate_expr(duplicate_expression_circuit: QuantumCircuit) -> None:
    circ = duplicate_expression_circuit
    assert len(circ.unique_parameters) == 4
    model_psr = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.GPSR)
    model_ad = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    assert len([i for i in model_psr.parameters()]) == 3
    assert len([i for i in model_ad.parameters()]) == 3
    embedded_params_psr = model_psr.embedding_fn(model_psr._params, {"x": torch.rand(1)})
    embedded_params_ad = model_ad.embedding_fn(model_ad._params, {"x": torch.rand(1)})
    assert len(embedded_params_ad) == 2 + 4  # adding 4 because all original params are included
    assert len(embedded_params_psr) == 8


def test_quantum_model_with_hevo() -> None:
    n_qubits = 4
    batch_size = 10

    # quantum circuit
    circuit = digital_analog_circ(n_qubits=n_qubits, depth=1)

    # random Hamiltonian matrices
    h = torch.rand(batch_size, 2**n_qubits, 2**n_qubits)
    hams = h + torch.conj(torch.transpose(h, 1, 2))
    values = {"gevo": hams}

    model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    wf = model.run(values)

    assert wf.size()[0] == batch_size


@pytest.mark.parametrize("n_qubits", [3, 4, 6])
def test_quantum_model_with_toffoli(n_qubits: int) -> None:
    prep_block = kron(X(i) for i in range(n_qubits))
    block = chain(prep_block, Toffoli(tuple(range(n_qubits - 1)), n_qubits - 1))
    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    wf = model.run({})
    assert wf[0][2 ** (n_qubits) - 2] == 1


@pytest.mark.parametrize("n_qubits", [3, 4, 6])
@pytest.mark.parametrize("gate", [MCRX, MCRX, MCRX])
def test_quantum_model_with_multi_controlled_rotation(gate: Any, n_qubits: int) -> None:
    prep_block = kron(X(i) for i in range(n_qubits))
    block = chain(prep_block, gate(tuple(range(n_qubits - 1)), n_qubits - 1, 2 * PI))
    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    wf = model.run({})
    assert wf[0][-1] == -1


def test_negative_scale_qm() -> None:
    hamilt = kron(Z(0), Z(1)) - 10 * Z(0)
    circ = QuantumCircuit(2, HamEvo(hamilt, 3))
    model = QuantumModel(circ, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    assert not torch.all(torch.isnan(model.run({})))


def test_save_load_qm_pyq(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    pyq_model = BasicQuantumModel
    for save_params in [True, False]:
        pyq_model.save(tmp_path, save_params=save_params)
        pyq_model_loaded = QuantumModel.load(tmp_path, save_params)
        pyq_expectation_orig = pyq_model.expectation({})[0]
        pyq_expectation_loaded = pyq_model_loaded.expectation({})[0]
        ser_qm = QuantumModel._from_dict(BasicQuantumModel._to_dict(save_params), save_params)
        ser_exp = ser_qm.expectation({})
        assert torch.allclose(ser_exp, pyq_expectation_orig)
        assert torch.allclose(pyq_expectation_orig, pyq_expectation_loaded)


def test_load_params_from_dict(BasicQuantumModel: QuantumModel) -> None:
    model = BasicQuantumModel
    ev0 = model.expectation({})[0]
    d = model._to_dict(save_params=True)
    model.load_params_from_dict(d, strict=True)
    ev1 = model.expectation({})[0]
    assert torch.allclose(ev0, ev1)

    # Check that an error is thrown if the dict does not match
    # the model parameters when strict=True
    d["param_dict"]["new_dummy_parameter"] = VariationalParameter("new_dummy_parameter")
    with pytest.raises(RuntimeError):
        model.load_params_from_dict(d, strict=True)

    # If strict=False, it should not throw an exception
    model.load_params_from_dict(d, strict=False)
    assert not torch.all(torch.isnan(model.expectation({})))


def test_hamevo_qm() -> None:
    obs = [Z(0) for _ in range(np.random.randint(1, 4))]
    block = HamEvo(VariationalParameter("theta") * X(1), 1, (0, 1))
    circ = QuantumCircuit(2, block)
    model = QuantumModel(circ, obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)  # type: ignore  # noqa
    assert not torch.all(torch.isnan(model.expectation({})))


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(BackendName.PULSER, marks=[pytest.mark.xfail]),
    ],
)
def test_correct_order(backend: BackendName) -> None:
    circ = QuantumCircuit(3, X(0))
    obs = [Z(0) for _ in range(np.random.randint(1, 5))]
    pyq_model = QuantumModel(
        circ, observable=obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD  # type: ignore
    )
    other_model = QuantumModel(
        circ, observable=obs, backend=backend, diff_mode=DiffMode.GPSR  # type: ignore
    )

    pyq_exp = pyq_model.expectation({})
    other_exp = other_model.expectation({})

    assert pyq_exp.size() == other_exp.size()
    assert torch.all(torch.isclose(pyq_exp, other_exp, atol=ATOL_DICT[BackendName.PULSER]))


def test_qc_obs_different_support_0() -> None:
    model_sup1 = QuantumModel(
        QuantumCircuit(1, RX(0, FeatureParameter("x"))),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    model_sup2 = QuantumModel(
        QuantumCircuit(2, RX(0, FeatureParameter("x"))),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )
    query_dict = {"x": torch.tensor([1.57])}
    assert torch.isclose(model_sup1.expectation(query_dict), model_sup2.expectation(query_dict))


@pytest.mark.parametrize("diff_mode", ["ad", "adjoint", "gpsr"])
def test_qc_obs_different_support_1(diff_mode: str) -> None:
    model_obs0_id_0 = QuantumModel(
        QuantumCircuit(1, I(0)),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
    )

    model_obs0_rot1 = QuantumModel(
        QuantumCircuit(2, RX(1, FeatureParameter("x"))),
        observable=Z(0),
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
    )

    model_obs01_rot1 = QuantumModel(
        QuantumCircuit(2, RX(1, FeatureParameter("x"))),
        observable=Z(0) + Z(1),
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
    )

    model_obs1_rot1 = QuantumModel(
        QuantumCircuit(2, RX(1, FeatureParameter("x"))),
        observable=I(0) + Z(1),
        backend=BackendName.PYQTORCH,
        diff_mode=diff_mode,
    )
    x = torch.tensor([2.1], requires_grad=True)
    query_dict = {"x": x}

    assert torch.isclose(model_obs0_rot1.expectation(query_dict), model_obs0_id_0.expectation({}))
    assert torch.isclose(
        model_obs01_rot1.expectation(query_dict), model_obs1_rot1.expectation(query_dict)
    )

    def fn(model: QuantumModel, x: torch.Tensor) -> torch.Tensor:
        return model.expectation({"x": x})

    for m in [model_obs0_rot1, model_obs1_rot1, model_obs01_rot1]:
        assert torch.autograd.gradcheck(lambda x: fn(m, x), x, nondet_tol=ADJOINT_ACCEPTANCE)


def test_distinct_obs_invert() -> None:
    qc = QuantumCircuit(2, chain(RX(0, FeatureParameter("x")), RX(1, FeatureParameter("y"))))
    obs = Z(0) + Z(1)

    qc_inv = invert_endianness(qc)
    obs_inv = invert_endianness(obs)

    m_pyq = QuantumModel(
        qc,
        obs,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )

    m_pyq_inv = QuantumModel(
        qc_inv,
        obs_inv,
        backend=BackendName.PYQTORCH,
        diff_mode=DiffMode.AD,
    )

    query_dict = {"x": torch.tensor([2.1]), "y": torch.tensor([2.1])}

    assert torch.isclose(m_pyq.expectation(query_dict), m_pyq_inv.expectation(query_dict))


def test_qm_obs_single_feature_param() -> None:
    cost_v = VariationalParameter("x") * Z(0)
    model_v = QuantumModel(
        QuantumCircuit(1, I(0)), cost_v, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )
    model_v.reset_vparams([2.7])
    model_v_exp = model_v.expectation({})
    cost_f = FeatureParameter("x") * Z(0)
    model_f = QuantumModel(
        QuantumCircuit(1, I(0)), cost_f, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )
    assert torch.all(torch.isclose(model_f.expectation({"x": torch.tensor([2.7])}), model_v_exp))


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "observables",
    [[FeatureParameter("x") * Z(0)], [FeatureParameter("x") * Z(0) for i in range(2)]],
)
def test_qm_obs_batch_feature_param(batch_size: int, observables: list[AbstractBlock]) -> None:
    n_obs = len(observables)
    random_batch = torch.rand(batch_size)
    batch_query_dict = {"x": random_batch}
    expected_output = random_batch.unsqueeze(1).repeat(1, n_obs)
    assert expected_output.shape == (batch_size, n_obs)
    model_f = QuantumModel(
        QuantumCircuit(1, I(0)), observables, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )
    model_f_exp = model_f.expectation(batch_query_dict)

    assert torch.all(torch.isclose(model_f_exp, expected_output))


def test_model_inputs_in_observable() -> None:
    w = FeatureParameter("w")
    m = QuantumModel(QuantumCircuit(1, X(0)), observable=w * Z(0))
    assert m.inputs == [w]


@given(st.restricted_circuits())
@settings(deadline=None)
def test_run_for_different_backends(circuit: QuantumCircuit) -> None:
    pyq_model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode="ad")
    inputs = rand_featureparameters(circuit, 1)
    inputs_jax = {k: tensor_to_jnp(v) for k, v in inputs.items()}

    bknd = backend_factory(BackendName.HORQRUX)
    (conv_circ, _, embed, params) = bknd.convert(circuit)
    wf_horqrux = bknd.run(conv_circ, embed(params, inputs_jax))

    assert equivalent_state(pyq_model.run(inputs), jarr_to_tensor(wf_horqrux))


@given(st.restricted_circuits())
@settings(deadline=None)
def test_sample_for_different_backends(circuit: QuantumCircuit) -> None:
    pyq_model = QuantumModel(circuit, backend=BackendName.PYQTORCH, diff_mode="ad")
    inputs = rand_featureparameters(circuit, 1)
    pyq_samples = pyq_model.sample(inputs, n_shots=100)

    inputs_jax = {k: tensor_to_jnp(v) for k, v in inputs.items()}

    bknd = backend_factory(BackendName.HORQRUX)
    (conv_circ, _, embed, params) = bknd.convert(circuit)
    horqrux_samples = bknd.sample(conv_circ, embed(params, inputs_jax), n_shots=100)

    for pyq_sample, sample in zip(pyq_samples, horqrux_samples):
        assert js_divergence(pyq_sample, sample) < JS_ACCEPTANCE + ATOL_DICT[BackendName.HORQRUX]


@given(st.restricted_circuits())
@settings(deadline=None)
def test_expectation_for_different_backends(circuit: QuantumCircuit) -> None:
    observable = [total_magnetization(circuit.n_qubits) for _ in range(np.random.randint(1, 5))]
    pyq_model = QuantumModel(
        circuit, observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )

    inputs = rand_featureparameters(circuit, 1)
    inputs_jax = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    pyq_expectation = pyq_model.expectation(inputs)

    bknd = backend_factory(BackendName.HORQRUX)
    (conv_circ, obs, embed, params) = bknd.convert(circuit, observable)
    horqrux_expectation = jarr_to_tensor(
        bknd.expectation(conv_circ, obs, embed(params, inputs_jax)), dtype=torch.double
    )

    assert torch.allclose(pyq_expectation, horqrux_expectation)


def test_to_pauli_list() -> None:

    qm = QuantumModel(QuantumCircuit(2, RX(1, FeatureParameter("x"))), observable=I(0) + Z(1))
    assert qm.to_pauli_list() == "Obs. : (I(0) + Z(1))"
