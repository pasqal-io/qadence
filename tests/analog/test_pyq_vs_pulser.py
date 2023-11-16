from __future__ import annotations

import numpy as np
import pytest
import torch
from metrics import ATOL_DICT, JS_ACCEPTANCE, LARGE_SPACING, SMALL_SPACING

from qadence.analog import RydbergDevice
from qadence.backends.pulser.devices import Device
from qadence.blocks import AbstractBlock, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import ising_hamiltonian, total_magnetization
from qadence.divergences import js_divergence
from qadence.models import QuantumModel
from qadence.operations import (
    CNOT,
    RX,
    RY,
    RZ,
    AnalogRot,
    AnalogRX,
    AnalogRY,
    AnalogRZ,
    H,
    X,
    Z,
    entangle,
    wait,
)
from qadence.parameters import FeatureParameter
from qadence.register import Register
from qadence.states import equivalent_state, random_state
from qadence.types import BackendName, DiffMode


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("spacing", [6.0, 8.0, 15.0])
@pytest.mark.parametrize("rydberg_level", [60, 85])
@pytest.mark.parametrize("op", [AnalogRX, AnalogRY, AnalogRZ, AnalogRot, wait])
def test_analog_op_run(
    n_qubits: int, spacing: float, rydberg_level: int, op: AbstractBlock
) -> None:
    init_state = random_state(n_qubits)
    batch_size = 3

    if op in [AnalogRX, AnalogRY, AnalogRZ]:
        phi = FeatureParameter("phi")
        block = op(phi)  # type: ignore [operator]
        values = {"phi": 1.0 + torch.rand(batch_size)}
    elif op == AnalogRot:
        t = 5.0
        omega = 1.0 + torch.rand(1)
        delta = 1.0 + torch.rand(1)
        phase = 1.0 + torch.rand(1)
        block = op(t, omega, delta, phase)  # type: ignore [operator]
        values = {}
    else:
        t = FeatureParameter("t")
        block = op(t)  # type: ignore [operator]
        values = {"t": 10.0 * (1.0 + torch.rand(batch_size))}

    register = Register.line(n_qubits, spacing=spacing)

    circuit = QuantumCircuit(register, block)

    device = RydbergDevice(register, rydberg_level=rydberg_level)

    config = {"device": device}

    model_pyqtorch = QuantumModel(
        circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, configuration=config
    )
    model_pyqtorch = QuantumModel(circuit, backend=BackendName.PYQTORCH)

    model_pulser = QuantumModel(
        circuit,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        configuration=config,
    )

    wf_pyq = model_pyqtorch.run(values=values, state=init_state)
    wf_pulser = model_pulser.run(values=values, state=init_state)

    assert equivalent_state(wf_pyq, wf_pulser, atol=ATOL_DICT[BackendName.PULSER])


# PREVIOUS COMPARISON TESTS, MOVED HERE


@pytest.mark.parametrize(
    "pyqtorch_block, pulser_block",
    [
        # Bell state generation
        (
            chain(H(0), CNOT(0, 1)),
            chain(entangle(1000, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2)),
        )
    ],
)
@pytest.mark.flaky(max_runs=5)
def test_compatibility_pyqtorch_pulser_entanglement(
    pyqtorch_block: AbstractBlock, pulser_block: AbstractBlock
) -> None:
    register = Register.line(2, spacing=8.0)

    pyqtorch_circuit = QuantumCircuit(register, pyqtorch_block)
    pulser_circuit = QuantumCircuit(register, pulser_block)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )
    config = {"device_type": Device.REALISTIC}
    model_pulser = QuantumModel(
        pulser_circuit, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR, configuration=config
    )
    pyqtorch_samples = model_pyqtorch.sample({}, n_shots=500)
    pulser_samples = model_pulser.sample({}, n_shots=500)
    for pyqtorch_sample, pulser_sample in zip(pyqtorch_samples, pulser_samples):
        assert js_divergence(pyqtorch_sample, pulser_sample) < JS_ACCEPTANCE


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("obs", [Z(0), total_magnetization(2), X(0), ising_hamiltonian(2)])
def test_compatibility_pyqtorch_pulser_digital_rot(obs: AbstractBlock) -> None:
    phi = FeatureParameter("phi")
    psi = FeatureParameter("psi")
    chi = FeatureParameter("chi")

    n_qubits = 2
    init_state = random_state(n_qubits)

    block = chain(
        kron(RX(0, phi), RX(1, phi)),
        kron(RY(0, psi), RY(1, psi)),
        kron(RZ(0, chi), RZ(1, chi)),
    )
    pyqtorch_circuit = QuantumCircuit(n_qubits, block)

    register = Register.line(n_qubits, spacing=LARGE_SPACING)
    pulser_circuit = QuantumCircuit(register, block)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )
    conf = {"amplitude_local": 2 * np.pi, "detuning": 2 * np.pi}

    model_pulser = QuantumModel(
        pulser_circuit,
        backend=BackendName.PULSER,
        observable=obs,
        diff_mode=DiffMode.GPSR,
        configuration=conf,
    )

    batch_size = 5
    values = {
        "phi": torch.rand(batch_size),
        "psi": torch.rand(batch_size),
        "chi": torch.rand(batch_size),
    }

    pyqtorch_expval = model_pyqtorch.expectation(values=values, state=init_state)
    pulser_expval = model_pulser.expectation(values=values, state=init_state)

    assert torch.allclose(pyqtorch_expval, pulser_expval, atol=ATOL_DICT[BackendName.PULSER])


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "obs",
    [
        Z(0),
        total_magnetization(2),
        X(0),
        ising_hamiltonian(2),
    ],
)
def test_compatibility_pyqtorch_pulser_analog_rot(obs: AbstractBlock) -> None:
    phi = FeatureParameter("phi")
    psi = FeatureParameter("psi")

    n_qubits = 2

    b_digital = chain(
        kron(RX(0, phi), RX(1, phi)),
        kron(RY(0, psi), RY(1, psi)),
    )

    b_analog = chain(AnalogRX(phi), AnalogRY(psi))
    pyqtorch_circuit = QuantumCircuit(n_qubits, b_digital)

    register = Register.line(n_qubits, spacing=LARGE_SPACING)
    pulser_circuit = QuantumCircuit(register, b_analog)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )

    model_pulser = QuantumModel(
        pulser_circuit,
        backend=BackendName.PULSER,
        observable=obs,
        diff_mode=DiffMode.GPSR,
    )

    batch_size = 5
    values = {
        "phi": torch.rand(batch_size),
        "psi": torch.rand(batch_size),
    }

    pyqtorch_expval = model_pyqtorch.expectation(values=values)
    pulser_expval = model_pulser.expectation(values=values)

    assert torch.allclose(pyqtorch_expval, pulser_expval, atol=ATOL_DICT[BackendName.PULSER])


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "obs",
    [
        Z(0),
        total_magnetization(2),
        X(0),
        ising_hamiltonian(2),
    ],
)
def test_compatibility_pyqtorch_pulser_analog_rot_int(obs: AbstractBlock) -> None:
    phi = FeatureParameter("phi")
    psi = FeatureParameter("psi")

    n_qubits = 2
    register = Register.line(n_qubits, spacing=SMALL_SPACING)

    b_analog = chain(AnalogRX(phi), AnalogRY(psi))

    circuit = QuantumCircuit(register, b_analog)

    model_pyqtorch = QuantumModel(
        circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )

    model_pulser = QuantumModel(
        circuit,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        observable=obs,
    )

    batch_size = 5
    values = {
        "phi": torch.rand(batch_size),
        "psi": torch.rand(batch_size),
    }

    pyqtorch_expval = model_pyqtorch.expectation(values=values)
    pulser_expval = model_pulser.expectation(values=values)

    assert torch.allclose(pyqtorch_expval, pulser_expval, atol=ATOL_DICT[BackendName.PULSER])
