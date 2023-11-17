from __future__ import annotations

import pytest
from metrics import ATOL_64

from qadence import (
    HamEvo,
    QuantumCircuit,
    QuantumModel,
    Register,
    daqc_transform,
    hamiltonian_factory,
    random_state,
)
from qadence.states import equivalent_state
from qadence.types import Interaction


@pytest.mark.parametrize("n_qubits", [2, 3, 5])
@pytest.mark.parametrize("t_f", [0.1, 10])
@pytest.mark.parametrize(
    "int_build, int_target",
    [
        (Interaction.ZZ, Interaction.ZZ),
        (Interaction.ZZ, Interaction.NN),
        (Interaction.NN, Interaction.ZZ),
        (Interaction.NN, Interaction.NN),
    ],
)
def test_daqc_ising(
    n_qubits: int,
    t_f: float,
    int_build: Interaction,
    int_target: Interaction,
) -> None:
    """Tests the DAQC transformation works for a random target and build hamiltonian."""
    gen_build = hamiltonian_factory(n_qubits, interaction=int_build, random_strength=True)
    gen_target = hamiltonian_factory(n_qubits, interaction=int_target, random_strength=True)

    transformed_circuit = daqc_transform(
        n_qubits=n_qubits,
        gen_target=gen_target,
        t_f=t_f,
        gen_build=gen_build,
    )

    circuit_daqc = QuantumCircuit(n_qubits, transformed_circuit)
    circuit_digital_block = QuantumCircuit(n_qubits, HamEvo(gen_target, t_f))
    model_digital = QuantumModel(circuit_digital_block)
    model_analog = QuantumModel(circuit_daqc)

    wf_init = random_state(n_qubits)

    wf_digital = model_digital.run(values={}, state=wf_init)
    wf_analog = model_analog.run(values={}, state=wf_init)

    assert equivalent_state(wf_digital, wf_analog, atol=10 * t_f * ATOL_64)


@pytest.mark.parametrize("n_qubits", [2, 3, 5])
@pytest.mark.parametrize("t_f", [0.1, 10])
@pytest.mark.parametrize(
    "int_build, int_target",
    [
        (Interaction.ZZ, Interaction.ZZ),
        (Interaction.ZZ, Interaction.NN),
        (Interaction.NN, Interaction.ZZ),
        (Interaction.NN, Interaction.NN),
    ],
)
def test_daqc_local(
    n_qubits: int,
    t_f: float,
    int_build: Interaction,
    int_target: Interaction,
) -> None:
    """Tests the DAQC transformation works for a local target hamiltonian.

    Uses a global random one.
    """
    gen_build = hamiltonian_factory(n_qubits, interaction=int_build, random_strength=True)
    register_target = Register.line(2)
    gen_target = hamiltonian_factory(register_target, interaction=int_target, random_strength=True)

    transformed_circuit = daqc_transform(
        n_qubits=n_qubits,
        gen_target=gen_target,
        t_f=t_f,
        gen_build=gen_build,
    )

    circuit_daqc = QuantumCircuit(n_qubits, transformed_circuit)
    circuit_digital_block = QuantumCircuit(n_qubits, HamEvo(gen_target, t_f))

    model_digital = QuantumModel(circuit_digital_block)
    model_analog = QuantumModel(circuit_daqc)

    wf_init = random_state(n_qubits)
    wf_digital = model_digital.run(values={}, state=wf_init)
    wf_analog = model_analog.run(values={}, state=wf_init)

    assert equivalent_state(wf_digital, wf_analog, atol=10 * t_f * ATOL_64)
