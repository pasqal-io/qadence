from __future__ import annotations

import numpy as np
import pytest
import torch
from metrics import ATOL_DICT, JS_ACCEPTANCE, LARGE_SPACING, SMALL_SPACING  # type: ignore

from qadence import BackendName, Register, add_interaction
from qadence.backends.pulser.devices import Device
from qadence.blocks import AbstractBlock, chain, kron
from qadence.circuit import QuantumCircuit
from qadence.constructors import ising_hamiltonian, total_magnetization
from qadence.divergences import js_divergence
from qadence.models import QuantumModel
from qadence.operations import CNOT, RX, RY, RZ, AnalogRX, AnalogRY, H, X, Z, entangle
from qadence.parameters import FeatureParameter
from qadence.states import random_state
from qadence.types import DiffMode


# "Compare" Pulser and PyQ
# NOTE: Since they are use different concepts, here only equivalent
# circuits/pulses are used.
@pytest.mark.parametrize(
    "pyqtorch_circuit,pulser_circuit",
    [
        # Bell state generation
        (
            QuantumCircuit(2, chain(H(0), CNOT(0, 1))),
            QuantumCircuit(2, chain(entangle(1000, qubit_support=(0, 1)), RY(0, 3 * torch.pi / 2))),
        )
    ],
)
@pytest.mark.flaky(max_runs=5)
def test_compatibility_pyqtorch_pulser_entanglement(
    pyqtorch_circuit: QuantumCircuit, pulser_circuit: QuantumCircuit
) -> None:
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

    register = Register.line(n_qubits)
    pulser_circuit = QuantumCircuit(register, block)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )
    conf = {"spacing": LARGE_SPACING, "amplitude_local": 2 * np.pi, "detuning": 2 * np.pi}
    model_pulser = QuantumModel(
        pulser_circuit,
        backend=BackendName.PULSER,
        observable=obs,
        diff_mode=DiffMode.GPSR,
        configuration=conf,
    )

    # TODO: Change batch_size back to 5 when respective `pyqtorch` bug is fixed:
    # https://github.com/pasqal-io/qadence/issues/148
    batch_size = 1
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

    register = Register.line(n_qubits)
    pulser_circuit = QuantumCircuit(register, b_analog)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )
    conf = {"spacing": LARGE_SPACING}
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
    register = Register.line(n_qubits)

    b_analog = chain(AnalogRX(phi), AnalogRY(psi))
    pyqtorch_circuit = QuantumCircuit(register, b_analog)
    pyqtorch_circuit = add_interaction(pyqtorch_circuit, spacing=SMALL_SPACING)

    pulser_circuit = QuantumCircuit(register, b_analog)

    model_pyqtorch = QuantumModel(
        pyqtorch_circuit, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD, observable=obs
    )
    conf = {"spacing": SMALL_SPACING}
    model_pulser = QuantumModel(
        pulser_circuit,
        backend=BackendName.PULSER,
        diff_mode=DiffMode.GPSR,
        observable=obs,
        configuration=conf,
    )

    batch_size = 5
    values = {
        "phi": torch.rand(batch_size),
        "psi": torch.rand(batch_size),
    }

    pyqtorch_expval = model_pyqtorch.expectation(values=values)
    pulser_expval = model_pulser.expectation(values=values)

    assert torch.allclose(pyqtorch_expval, pulser_expval, atol=ATOL_DICT[BackendName.PULSER])
