from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import sympy
import torch
from metrics import AGPSR_ACCEPTANCE, GPSR_ACCEPTANCE, PSR_ACCEPTANCE

from qadence import DiffMode, Parameter, QuantumCircuit
from qadence.analog import add_background_hamiltonian
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, add, chain
from qadence.constructors import total_magnetization
from qadence.engines.torch.differentiable_backend import DifferentiableBackend
from qadence.execution import expectation
from qadence.operations import CNOT, CRX, CRY, RX, RY, AnalogRot, HamEvo, X, Y, Z, N
from qadence.register import Register
from qadence.types import PI


def circuit_psr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    fm = chain(RX(0, 3 * x), RY(1, sympy.exp(x)), RX(0, theta), RY(1, PI / 2))
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_gpsr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    fm = chain(
        CRX(0, 1, 3 * x),
        X(1),
        CRY(1, 2, sympy.exp(x)),
        CRX(1, 2, theta),
        X(0),
        CRY(0, 1, PI / 2),
    )
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_hamevo_tensor_gpsr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    h = torch.rand(2**n_qubits, 2**n_qubits)
    ham = h + torch.conj(torch.transpose(h, 0, 1))
    ham = ham[None, :, :]

    fm = chain(
        CRX(0, 1, 3 * x),
        X(1),
        CRY(1, 2, sympy.exp(x)),
        HamEvo(ham, x, qubit_support=tuple(range(n_qubits))),
        CRX(1, 2, theta),
        X(0),
        CRY(0, 1, PI / 2),
    )
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_hamevo_tensor_agpsr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)

    h = torch.rand(2**n_qubits, 2**n_qubits)
    ham = h + torch.conj(torch.transpose(h, 0, 1))
    ham = ham[None, :, :]

    fm = chain(
        CRY(1, 2, sympy.exp(x)),
        HamEvo(ham, x, qubit_support=tuple(range(n_qubits))),
        X(0),
        CRY(0, 1, np.pi / 2),
    )
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_hamevo_block_gpsr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    dim = np.random.randint(1, n_qubits + 1)
    ops = [X, Y, Z] * 2
    qubit_supports = np.random.choice(dim, len(ops), replace=True)
    generator = chain(
        add(*[op(q) for op, q in zip(ops, qubit_supports)]),  # type: ignore [abstract]
        *[op(q) for op, q in zip(ops, qubit_supports)],  # type: ignore [abstract]
    )
    generator = generator + generator.dagger()  # type: ignore [assignment]

    fm = chain(
        CRX(0, 1, 3 * x),
        X(1),
        CRY(1, 2, sympy.exp(x)),
        HamEvo(generator, x, qubit_support=tuple(range(n_qubits))),
        CRX(1, 2, theta),
        X(0),
        CRY(0, 1, PI / 2),
    )
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_analog_rotation_gpsr(n_qubits: int) -> QuantumCircuit:
    d = 10
    omega1 = 6 * PI
    omega2 = 3 * PI
    coords = [(x_coord, 0) for x_coord in np.linspace(0, (n_qubits - 1) * d, n_qubits)]
    register = Register.from_coordinates(coords)  # type: ignore[arg-type]

    # circuit with builting primitives
    x = Parameter("x", trainable=False)
    theta = Parameter("theta")
    analog_block = chain(
        AnalogRot(duration=1000 * x / omega1, omega=omega1, delta=0, phase=0),
        AnalogRot(duration=1000 * theta / omega2, omega=omega2, delta=0, phase=0),
    )

    circ = QuantumCircuit(register, analog_block)

    return add_background_hamiltonian(circ)  # type: ignore [return-value]


sum_N: Callable = lambda n: sum(i * N(i) for i in range(n))


@pytest.mark.parametrize(
    ["n_qubits", "batch_size", "n_obs", "circuit_fn"],
    [
        (2, 1, 2, circuit_psr),
        (5, 10, 1, circuit_psr),
        (3, 1, 4, circuit_gpsr),
        (5, 10, 1, circuit_gpsr),
        (3, 1, 1, circuit_hamevo_tensor_gpsr),
        (3, 1, 1, circuit_hamevo_block_gpsr),
        (3, 1, 1, circuit_analog_rotation_gpsr),
    ],
)
@pytest.mark.parametrize("obs_fn", [total_magnetization, sum_N])
def test_expectation_psr(
    n_qubits: int, batch_size: int, n_obs: int, circuit_fn: Callable, obs_fn: Callable
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # Making circuit with AD
    circ = circuit_fn(n_qubits)
    obs = obs_fn(n_qubits)
    quantum_backend = PyQBackend()
    conv = quantum_backend.convert(circ, [obs for _ in range(n_obs)])
    pyq_circ, pyq_obs, embedding_fn, params = conv
    diff_backend = DifferentiableBackend(quantum_backend, diff_mode=DiffMode.AD)

    # Running for some inputs
    values = {"x": torch.rand(batch_size, requires_grad=True)}
    expval = diff_backend.expectation(pyq_circ, pyq_obs, embedding_fn(params, values))
    dexpval_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]

    dexpval_xx = torch.autograd.grad(
        dexpval_x, values["x"], torch.ones_like(dexpval_x), create_graph=True
    )[0]
    if circuit_fn not in [
        circuit_hamevo_tensor_gpsr,
        circuit_hamevo_block_gpsr,
        circuit_analog_rotation_gpsr,
    ]:
        dexpval_xxtheta = torch.autograd.grad(
            dexpval_xx,
            list(params.values())[0],
            torch.ones_like(dexpval_xx),
            retain_graph=True,
        )[0]
    dexpval_theta = torch.autograd.grad(expval, list(params.values())[0], torch.ones_like(expval))[
        0
    ]

    # Now running stuff for (G)PSR
    quantum_backend.config._use_gate_params = True
    conv = quantum_backend.convert(circ, [obs for _ in range(n_obs)])
    pyq_circ, pyq_obs, embedding_fn, params = conv
    if circuit_fn == circuit_analog_rotation_gpsr:
        diff_backend = DifferentiableBackend(
            quantum_backend, diff_mode=DiffMode.GPSR, shift_prefac=0.2
        )
    else:
        diff_backend = DifferentiableBackend(
            quantum_backend, diff_mode=DiffMode.GPSR, shift_prefac=0.2
        )
    expval = diff_backend.expectation(pyq_circ, pyq_obs, embedding_fn(params, values))
    dexpval_psr_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]

    dexpval_psr_xx = torch.autograd.grad(
        dexpval_psr_x, values["x"], torch.ones_like(dexpval_psr_x), create_graph=True
    )[0]
    if circuit_fn not in [
        circuit_hamevo_tensor_gpsr,
        circuit_hamevo_block_gpsr,
        circuit_analog_rotation_gpsr,
    ]:
        dexpval_psr_xxtheta = torch.autograd.grad(
            dexpval_psr_xx,
            list(params.values())[0],
            torch.ones_like(dexpval_psr_xx),
            retain_graph=True,
        )[0]
    dexpval_psr_theta = torch.autograd.grad(
        expval, list(params.values())[0], torch.ones_like(expval)
    )[0]

    atol = PSR_ACCEPTANCE if circuit_fn == circuit_psr else GPSR_ACCEPTANCE
    assert torch.allclose(dexpval_x, dexpval_psr_x, atol=atol), "df/dx not equal."
    assert torch.allclose(dexpval_xx, dexpval_psr_xx, atol=atol), " d2f/dx2 not equal."
    assert torch.allclose(dexpval_theta, dexpval_psr_theta, atol=atol), "df/dtheta not equal."
    if circuit_fn not in [
        circuit_hamevo_tensor_gpsr,
        circuit_hamevo_block_gpsr,
        circuit_analog_rotation_gpsr,
    ]:
        assert torch.allclose(
            dexpval_xxtheta, dexpval_psr_xxtheta, atol=atol
        ), "d3f/dx2dtheta not equal."


@pytest.mark.parametrize(
    ["n_qubits", "generator"],
    [
        (1, 0.5 * X(0)),
        (1, X(0)),
        (2, X(0) + Y(1)),
        (3, X(0) + 0.5 * Z(2)),
        (3, 10 * (X(0) + 0.5 * Z(2))),
    ],
)
def test_hamevo_gpsr(n_qubits: int, generator: AbstractBlock) -> None:
    x = Parameter("x", trainable=False)
    hamevo_block = HamEvo(generator, x)

    xs = torch.linspace(0, 2 * torch.pi, 100, requires_grad=True)
    values = {"x": xs}

    # Calculate function f(x)
    obs = add(Z(i) for i in range(n_qubits))
    exp_ad = expectation(hamevo_block, observable=obs, values=values, diff_mode=DiffMode.AD)
    exp_gpsr = expectation(hamevo_block, observable=obs, values=values, diff_mode=DiffMode.GPSR)

    # Check we are indeed computing the same thing
    assert torch.allclose(exp_ad, exp_gpsr)

    # calculate derivative df/dx using the PyTorch
    dfdx_ad = torch.autograd.grad(exp_ad, xs, torch.ones_like(exp_ad), create_graph=True)[0]
    dfdx_gpsr = torch.autograd.grad(exp_gpsr, xs, torch.ones_like(exp_gpsr), create_graph=True)[0]

    assert torch.allclose(dfdx_ad, dfdx_gpsr, atol=GPSR_ACCEPTANCE)


@pytest.mark.parametrize(
    ["n_qubits", "batch_size", "circuit_fn", "shift_prefac", "n_eqs", "lb", "ub"],
    [
        (3, 1, circuit_hamevo_tensor_agpsr, 0.5, 10, 0.1, 1.0),
        (3, 1, circuit_hamevo_tensor_agpsr, None, 10, 0.01, 0.6),
    ],
)
def test_expectation_agpsr(
    n_qubits: int,
    batch_size: int,
    circuit_fn: Callable,
    shift_prefac: float | None,
    n_eqs: int,
    lb: float,
    ub: float,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # Making circuit with AD
    circ = circuit_fn(n_qubits)
    obs = total_magnetization(n_qubits)

    # Running for some inputs
    values = {"x": torch.rand(batch_size, requires_grad=True)}
    expval = expectation(circ, observable=obs, values=values, diff_mode=DiffMode.AD)
    dexpval_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]
    dexpval_xx = torch.autograd.grad(
        dexpval_x, values["x"], torch.ones_like(dexpval_x), create_graph=True
    )[0]

    # Now running stuff for aGPSR
    config = {
        "shift_prefac": shift_prefac,
        "n_eqs": n_eqs,
        "lb": lb,
        "ub": ub,
    }
    expval = expectation(
        circ, observable=obs, values=values, diff_mode=DiffMode.GPSR, configuration=config
    )
    dexpval_psr_x = torch.autograd.grad(
        expval, values["x"], torch.ones_like(expval), create_graph=True
    )[0]
    dexpval_psr_xx = torch.autograd.grad(
        dexpval_psr_x, values["x"], torch.ones_like(dexpval_psr_x), create_graph=True
    )[0]

    assert torch.allclose(dexpval_x, dexpval_psr_x, atol=AGPSR_ACCEPTANCE), "df/dx not equal."
    assert torch.allclose(
        dexpval_xx, dexpval_psr_xx, atol=np.sqrt(AGPSR_ACCEPTANCE)
    ), " d2f/dx2 not equal."
