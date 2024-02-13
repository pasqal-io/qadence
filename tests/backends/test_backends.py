from __future__ import annotations

from typing import Counter

import numpy as np
import pytest
import strategies as st  # type: ignore
import sympy
import torch
from horqrux.utils import equivalent_state as horq_equivalent_state
from hypothesis import given, settings
from jax import Array
from metrics import ATOL_DICT, JS_ACCEPTANCE  # type: ignore
from torch import Tensor

from qadence import Interaction, Register, hamiltonian_factory, hea
from qadence.backend import BackendConfiguration
from qadence.backends.api import backend_factory, config_factory
from qadence.backends.jax_utils import jarr_to_tensor, tensor_to_jnp
from qadence.blocks import AbstractBlock, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import total_magnetization
from qadence.divergences import js_divergence
from qadence.execution import run
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QuantumModel
from qadence.operations import CPHASE, RX, RY, H, HamEvo, I, X, Y, Z
from qadence.parameters import FeatureParameter, Parameter
from qadence.states import (
    equivalent_state,
    product_state,
    rand_product_state,
    random_state,
    zero_state,
)
from qadence.transpile import flatten, set_trainable
from qadence.types import PI, BackendName, DiffMode
from qadence.utils import nqubits_to_basis

BACKENDS = BackendName.list()
BACKENDS.remove("pulser")


def flatten_counter(c: Counter | list[Counter]) -> Counter:
    if isinstance(c, Counter):
        sorted_counter = Counter(dict(sorted(c.items())))
        return sorted_counter

    elif isinstance(c, list):
        flattened_counter: Counter = Counter()
        for counter in c:
            flattened_counter += counter
        sorted_counter = Counter(dict(sorted(flattened_counter.items())))
        return sorted_counter

    else:
        raise TypeError("Input must be either a Counter object or a list of Counter objects.")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "circuit",
    [
        QuantumCircuit(1),
        QuantumCircuit(2, chain(X(0), X(1))),
        QuantumCircuit(3, chain(I(0), X(1), I(2))),
        QuantumCircuit(3, chain(X(1))),
    ],
)
def test_simple_circuits(backend: str, circuit: QuantumCircuit) -> None:
    bknd = backend_factory(backend=backend)
    wf = bknd.run(bknd.circuit(circuit))
    assert isinstance(wf, (Tensor, Array))


def test_expectation_value(parametric_circuit: QuantumCircuit) -> None:
    observable = total_magnetization(parametric_circuit.n_qubits)

    values = {"x": torch.rand(1)}
    wfs = []
    for b in BACKENDS:
        bkd = backend_factory(backend=b, diff_mode=None)
        conv = bkd.convert(parametric_circuit, observable)
        if b == BackendName.HORQRUX:
            values = {k: tensor_to_jnp(v) for k, v in values.items()}
        expval = bkd.expectation(
            conv.circuit, conv.observable, conv.embedding_fn(conv.params, values)  # type: ignore
        )
        wf = bkd.run(conv.circuit, conv.embedding_fn(conv.params, values))
        wfs.append(wf.flatten().sum())

    # normalize the type of the wavefunction
    wfs_np = []
    for wf in wfs:
        wfs_np.append(complex(wf))
    wfs_np = np.array(wfs_np)  # type: ignore [assignment]
    assert np.all(np.isclose(wfs_np, wfs_np[0]))


@pytest.mark.parametrize(
    "backend",
    [
        BackendName.PYQTORCH,
        BackendName.BRAKET,
        pytest.param(
            BackendName.HORQRUX,
            marks=pytest.mark.xfail(reason="Horqrux uses JAX engine."),
        ),
    ],
)
def test_qcl_loss(backend: str) -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    def get_training_data(domain: tuple = (-0.99, 0.99), n_teacher: int = 30) -> tuple:
        start, end = domain
        x_rand_np = np.sort(np.random.uniform(low=start, high=end, size=n_teacher))
        y_rand_np = x_rand_np * x_rand_np
        x_rand = torch.tensor(x_rand_np, requires_grad=True)
        y_rand = torch.tensor(y_rand_np, requires_grad=True)
        return x_rand, y_rand

    n_qubits = 2  # number of qubits in the circuit

    param = FeatureParameter("phi")
    featuremap = kron(RY(qubit, sympy.asin(param)) for qubit in range(n_qubits))
    circuit = QuantumCircuit(n_qubits, featuremap)
    observable = kron(X(i) for i in range(n_qubits))

    x_train, y_train = get_training_data(n_teacher=1)
    input_values = {"phi": x_train}

    # test expectation == 0
    model = QuantumModel(circuit, backend=BackendName(backend), diff_mode=DiffMode.GPSR)
    native_observable = model.observable(observable, n_qubits)
    e = model.expectation(input_values, native_observable)
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(e, y_train)
    assert torch.allclose(loss, torch.tensor(0.0))

    # test derivative of expectation == 2x
    d = torch.autograd.grad(e, x_train, torch.ones_like(e))[0]
    assert torch.allclose(d, 2 * x_train)


@pytest.mark.parametrize(
    "backend",
    [
        BackendName.PYQTORCH,
        pytest.param(
            BackendName.HORQRUX,
            marks=pytest.mark.xfail(reason="horqrux doesnt support batching of states."),
        ),
        pytest.param(
            BackendName.BRAKET,
            marks=pytest.mark.xfail(reason="state-vector initial state not implemented in Braket"),
        ),
    ],
)
def test_custom_initial_state(backend: str) -> None:
    circ = QuantumCircuit(2, chain(X(0), X(1)))
    bkd = backend_factory(backend)
    conv = bkd.convert(circ)

    # test single sample batch
    for input_state, target_state in zip(["01", "10", "11"], ["10", "01", "00"]):
        wf = product_state(input_state)  # type: ignore[arg-type]
        # need to use pyqtorch to construct 00 state
        target_wf = product_state(target_state)  # type: ignore[arg-type]
        assert equivalent_state(bkd.run(conv.circuit, state=wf), target_wf)

    # test batch
    wf = product_state(["01", "10", "11"])  # type: ignore[arg-type]
    assert equivalent_state(
        bkd.run(conv.circuit, state=wf), product_state(["10", "01", "00"])  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "circ", [QuantumCircuit(2, chain(X(0), X(1))), QuantumCircuit(2, chain(H(0), H(1)))]
)
@pytest.mark.flaky(max_runs=5)
def test_backend_sampling(circ: QuantumCircuit) -> None:
    bknd_pyqtorch = backend_factory(BackendName.PYQTORCH)
    bknd_braket = backend_factory(BackendName.BRAKET)

    (circ_pyqtorch, _, _, _) = bknd_pyqtorch.convert(circ)
    (circ_braket, _, embed, params) = bknd_braket.convert(circ)

    # braket doesn't support custom initial states, so we use state=None for the zero state
    pyqtorch_samples = bknd_pyqtorch.sample(
        circ_pyqtorch, embed(params, {}), state=None, n_shots=100
    )
    braket_samples = bknd_braket.sample(
        circ_braket,
        embed(params, {}),
        state=None,
        n_shots=100,
    )

    for pyqtorch_sample, braket_sample in zip(pyqtorch_samples, braket_samples):
        assert js_divergence(pyqtorch_sample, braket_sample) < JS_ACCEPTANCE

    wf_braket = bknd_braket.run(circ_braket)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch)
    assert equivalent_state(wf_braket, wf_pyqtorch, atol=ATOL_DICT[BackendName.BRAKET])


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_run_for_random_circuit(backend: BackendName, circuit: QuantumCircuit) -> None:
    cfg = {"_use_gate_params": True}
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ_pyqtorch, _, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit)
    (circ, _, embed, params) = bknd.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs))
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    wf = bknd.run(circ, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        wf = jarr_to_tensor(wf)
    assert equivalent_state(wf_pyqtorch, wf, atol=ATOL_DICT[backend])


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.flaky(max_runs=5)
def test_sample_for_random_circuit(backend: BackendName, circuit: QuantumCircuit) -> None:
    cfg = {"_use_gate_params": True}
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ_pyqtorch, _, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(circuit)
    (circ, _, embed, params) = bknd.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    pyqtorch_samples = bknd_pyqtorch.sample(
        circ_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs), n_shots=100
    )
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    samples = bknd.sample(circ, embed(params, inputs), n_shots=100)

    for pyqtorch_sample, sample in zip(pyqtorch_samples, samples):
        assert js_divergence(pyqtorch_sample, sample) < JS_ACCEPTANCE + ATOL_DICT[backend]


# TODO: include many observables
@given(st.restricted_circuits(), st.observables())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
def test_expectation_for_random_circuit(
    backend: BackendName, circuit: QuantumCircuit, observable: AbstractBlock
) -> None:
    if observable.n_qubits > circuit.n_qubits:
        circuit = QuantumCircuit(observable.n_qubits, circuit.block)
    cfg = {"_use_gate_params": True}
    bknd_pyqtorch = backend_factory(backend=BackendName.PYQTORCH, configuration=cfg)
    bknd = backend_factory(backend=backend, configuration=cfg)
    (circ_pyqtorch, obs_pyqtorch, embed_pyqtorch, params_pyqtorch) = bknd_pyqtorch.convert(
        circuit, observable
    )
    (circ, obs, embed, params) = bknd.convert(circuit, observable)
    inputs = rand_featureparameters(circuit, 1)
    pyqtorch_expectation = bknd_pyqtorch.expectation(
        circ_pyqtorch, obs_pyqtorch, embed_pyqtorch(params_pyqtorch, inputs)
    )[0]
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}

    expectation = bknd.expectation(circ, obs, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        expectation = jarr_to_tensor(expectation, dtype=torch.double)
    assert torch.allclose(pyqtorch_expectation, expectation, atol=ATOL_DICT[backend])


@given(st.restricted_circuits())
@settings(deadline=None)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.flaky(max_runs=5)
def test_compare_run_to_sample(backend: BackendName, circuit: QuantumCircuit) -> None:
    bknd = backend_factory(backend)
    (conv_circ, _, embed, params) = bknd.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    if inputs and backend == BackendName.HORQRUX:
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    samples = bknd.sample(conv_circ, embed(params, inputs), n_shots=1000)
    wf = bknd.run(conv_circ, embed(params, inputs))
    if backend == BackendName.HORQRUX:
        wf = jarr_to_tensor(wf)
    probs = list(torch.abs(torch.pow(wf, 2)).flatten().detach().numpy())
    bitstrngs = nqubits_to_basis(circuit.n_qubits)
    wf_counter = Counter(
        {bitstring: prob for (bitstring, prob) in zip(bitstrngs, probs) if prob > 0.0}
    )
    assert js_divergence(samples[0], wf_counter) < JS_ACCEPTANCE + ATOL_DICT[backend]


def test_default_configuration() -> None:
    for b in BACKENDS + [BackendName.PULSER]:
        bknd = backend_factory(b, diff_mode=None)
        conf = bknd.default_configuration()
        assert isinstance(conf, BackendConfiguration)
        opts = conf.available_options()
        assert isinstance(opts, str)


@given(st.digital_circuits())
@settings(deadline=None)
@pytest.mark.parametrize(
    "backend",
    [
        BackendName.PYQTORCH,
        pytest.param(
            BackendName.BRAKET,
            marks=pytest.mark.xfail(reason="Braket doesnt support passing custom states"),
        ),
    ],
)
def test_run_for_random_state(backend: str, circuit: QuantumCircuit) -> None:
    bknd_pyqtorch = backend_factory(backend)
    pyqtorch_state = random_state(circuit.n_qubits, backend=backend)
    rand_bit_state = rand_product_state(circuit.n_qubits)
    (circ_pyqtorch, _, embed, params) = bknd_pyqtorch.convert(circuit)
    inputs = rand_featureparameters(circuit, 1)
    embedded_params = embed(params, inputs)
    wf_pyqtorch = bknd_pyqtorch.run(circ_pyqtorch, embedded_params, pyqtorch_state)
    wf_randbit = bknd_pyqtorch.run(circ_pyqtorch, embedded_params, rand_bit_state)
    assert not torch.any(torch.isnan(wf_pyqtorch))
    assert not torch.any(torch.isnan(wf_randbit))


@pytest.mark.parametrize("bsize", [i for i in range(1, 10, 2)])
def test_output_cphase_batching(bsize: int) -> None:
    backend_list = [BackendName.BRAKET, BackendName.PYQTORCH]

    n_qubits = 4
    w = FeatureParameter("w")

    # Circuit is created here.
    circuit = QuantumCircuit(n_qubits, chain(X(0), CPHASE(1, 0, w), CPHASE(2, 1, w), RX(1, "x")))
    values = {"w": torch.rand(bsize)}
    exp_list = []
    wf_list = []
    for backend_name in backend_list:
        backend = backend_factory(backend_name)
        observable = [total_magnetization(n_qubits=circuit.n_qubits)] * 1
        (circ, obs, embed, params) = backend.convert(circuit, observable)

        val = embed(params, values)
        wf = backend.run(circ, val)
        wf_list.append(wf)

        expected = zero_state(n_qubits=4, batch_size=10)
        expected[0] = 1.0

        exp_list.append(backend.expectation(circ, obs, val))

    assert torch.allclose(exp_list[0], exp_list[1])
    assert equivalent_state(wf_list[0], wf_list[1])


def test_custom_transpilation_passes() -> None:
    backend_list = [BackendName.BRAKET, BackendName.PYQTORCH, BackendName.PULSER]

    block = chain(chain(chain(RX(0, PI / 2))), kron(kron(RX(0, PI / 2))))
    circuit = QuantumCircuit(1, block)

    for name in backend_list:
        config = config_factory(name, {})
        config.transpilation_passes = [flatten]
        backend = backend_factory(name, configuration=config)
        conv = backend.convert(circuit)

        config = config_factory(name, {})
        config.transpilation_passes = []
        backend_no_transp = backend_factory(name, configuration=config)
        conv_no_transp = backend_no_transp.convert(circuit)

        assert conv.circuit.original == conv_no_transp.circuit.original
        assert conv.circuit.abstract != conv_no_transp.circuit.abstract


def test_braket_parametric_cphase() -> None:
    param_name = "y"
    block = chain(X(0), H(1), CPHASE(0, 1, param_name))
    values = {param_name: torch.rand(1)}
    equivalent_state(
        run(block, values=values, backend="braket"), run(block, values=values, backend="pyqtorch")
    )


@pytest.mark.parametrize("backend_name", [BackendName.PYQTORCH, BackendName.HORQRUX])
def test_dagger_returning_kernel(backend_name: BackendName) -> None:
    def wf_is_normalized(wf: torch.Tensor) -> torch.Tensor:
        return torch.isclose(sum(torch.flatten(torch.abs(wf) ** 2)), torch.tensor(1.00))

    generatorx = 3.1 * X(0) + 1.2 * Y(0) + 1.1 * Y(1) + 1.9 * X(1) + 2.4 * Z(0) * Z(1)
    fmx = HamEvo(generatorx, parameter=sympy.acos(Parameter("x")))
    set_trainable(fmx, False)
    fmy = HamEvo(generatorx, parameter=sympy.acos(Parameter("y")))
    set_trainable(fmy, False)
    ansatz = hea(2, 2)
    set_trainable(ansatz, True)
    circ = QuantumCircuit(2, fmx, ansatz.dagger(), ansatz, fmy.dagger())
    backend = backend_factory(backend=backend_name, diff_mode=DiffMode.AD)
    (pyqtorch_circ, _, embed, params) = backend.convert(circ)

    initial_state = torch.rand((1, 2**2), dtype=torch.cdouble) + 1j * torch.rand(
        (1, 2**2), dtype=torch.cdouble
    )
    initial_state = initial_state / torch.sqrt(4 * sum(abs(initial_state) ** 2))
    inputs = {"x": torch.tensor([0.52]), "y": torch.tensor(0.52)}
    if backend_name == BackendName.HORQRUX:
        initial_state = tensor_to_jnp(initial_state)
        inputs = {k: tensor_to_jnp(v) for k, v in inputs.items()}
    run_params = embed(params, inputs)
    wf = backend.run(pyqtorch_circ, run_params, state=initial_state)
    if backend_name == BackendName.HORQRUX:
        assert horq_equivalent_state(wf, initial_state)
    else:
        assert wf_is_normalized(wf)
        assert equivalent_state(wf, initial_state)


@pytest.mark.parametrize("interaction", [Interaction.XY, Interaction.ZZ])
@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_compare_hevos(interaction: Interaction, n_qubits: int) -> None:
    register = Register.line(n_qubits)

    gen = hamiltonian_factory(
        register,
        interaction=interaction,
        random_strength=True,
        use_all_node_pairs=True,
    )

    t_evo = torch.rand(1)
    op = HamEvo(gen, t_evo)
    init_state_torch = random_state(n_qubits)
    init_state_jax = tensor_to_jnp(init_state_torch)
    wf_torch = run(op, state=init_state_torch, backend=BackendName.PYQTORCH)
    wf_jax = jarr_to_tensor(run(op, state=init_state_jax, backend=BackendName.HORQRUX))
    assert equivalent_state(wf_torch, wf_jax)
