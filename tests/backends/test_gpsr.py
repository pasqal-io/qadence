from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import sympy
import torch
from metrics import GPSR_ACCEPTANCE, PSR_ACCEPTANCE

from qadence import DifferentiableBackend, DiffMode, Parameter, QuantumCircuit
from qadence.analog import add_interaction
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import add, chain
from qadence.constructors import total_magnetization
from qadence.operations import CNOT, CRX, CRY, RX, RY, ConstantAnalogRotation, HamEvo, X, Y, Z
from qadence.parameters import ParamMap
from qadence.register import Register


def circuit_psr(n_qubits: int) -> QuantumCircuit:
    """Helper function to make an example circuit."""

    x = Parameter("x", trainable=False)
    theta = Parameter("theta")

    fm = chain(RX(0, 3 * x), RY(1, sympy.exp(x)), RX(0, theta), RY(1, np.pi / 2))
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
        CRY(0, 1, np.pi / 2),
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
        CRY(0, 1, np.pi / 2),
    )
    ansatz = CNOT(0, 1)
    block = chain(fm, ansatz)

    circ = QuantumCircuit(n_qubits, block)

    return circ


def circuit_analog_rotation_gpsr(n_qubits: int) -> QuantumCircuit:
    d = 10
    omega1 = 6 * np.pi
    omega2 = 3 * np.pi
    coords = [(x_coord, 0) for x_coord in np.linspace(0, (n_qubits - 1) * d, n_qubits)]
    register = Register.from_coordinates(coords)  # type: ignore[arg-type]

    # circuit with builting primitives
    x = Parameter("x", trainable=False)
    theta = Parameter("theta")
    analog_block = chain(
        ConstantAnalogRotation(
            parameters=ParamMap(duration=1000 * x / omega1, omega=omega1, delta=0, phase=0)
        ),
        ConstantAnalogRotation(
            parameters=ParamMap(duration=1000 * theta / omega2, omega=omega2, delta=0, phase=0)
        ),
    )

    block = add_interaction(register, analog_block).block  # type: ignore [arg-type]
    circ = QuantumCircuit(n_qubits, block)

    return circ


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
def test_expectation_psr(n_qubits: int, batch_size: int, n_obs: int, circuit_fn: Callable) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # Making circuit with AD
    circ = circuit_fn(n_qubits)
    obs = total_magnetization(n_qubits)
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
