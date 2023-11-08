from __future__ import annotations

import pytest
import torch
from metrics import ATOL_64

from qadence import (
    BackendName,
    Interaction,
    QuantumCircuit,
    QuantumModel,
    hamiltonian_factory,
    qft,
    random_state,
)
from qadence.states import equivalent_state
from qadence.types import Strategy


def test_qft() -> None:
    def qft_matrix(N: int) -> torch.Tensor:
        """Textbook QFT unitary matrix to compare to the circuit solution."""
        matrix = torch.zeros((N, N), dtype=torch.cdouble)
        w = torch.exp(torch.tensor(2.0j * torch.pi / N, dtype=torch.cdouble))
        for i in range(N):
            for j in range(N):
                matrix[i, j] = (N ** (-1 / 2)) * w ** (i * j)
        return matrix

    n_qubits = 2

    # First tests that the qft_matrix function is correct for 2-qubits
    qft_m_2q = (1 / 2) * torch.tensor(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j],
            [1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 - 1.0j, -1.0 + 0.0j, 0.0 + 1.0j],
        ],
        dtype=torch.cdouble,
    )

    assert torch.allclose(qft_m_2q, qft_matrix(n_qubits**2), rtol=0.0, atol=ATOL_64)

    # Now loads larger random initial state
    n_qubits = 5

    wf_init = random_state(n_qubits)

    # Runs QFT circuit with swaps to match standard QFT definition
    qc_qft = QuantumCircuit(n_qubits, qft(n_qubits, swaps_out=True, strategy=Strategy.DIGITAL))
    model = QuantumModel(qc_qft, backend=BackendName.PYQTORCH)
    wf_qft = model.run(values={}, state=wf_init)

    # Checks output with the textbook matrix
    wf_textbook = torch.matmul(qft_matrix(2**n_qubits), wf_init[0])

    assert equivalent_state(wf_qft, wf_textbook.unsqueeze(0), atol=10 * ATOL_64)


def test_qft_inverse() -> None:
    """Tests that applying qft -> inverse qft returns the initial state."""
    n_qubits = 4
    wf_init = random_state(n_qubits)
    qc_qft = QuantumCircuit(n_qubits, qft(n_qubits))
    qc_qft_inv = QuantumCircuit(n_qubits, qft(n_qubits, inverse=True))
    model = QuantumModel(qc_qft, backend=BackendName.PYQTORCH)
    model_inv = QuantumModel(qc_qft_inv, backend=BackendName.PYQTORCH)
    wf_1 = model.run(values={}, state=wf_init)
    wf_2 = model_inv.run(values={}, state=wf_1)
    assert equivalent_state(wf_2, wf_init, atol=ATOL_64)


@pytest.mark.parametrize(
    "param_dict",
    [
        {"inverse": False, "reverse_in": False, "swaps_out": False},
        {"inverse": True, "reverse_in": True, "swaps_out": True},
    ],
)
@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_qft_digital_analog(n_qubits: int, param_dict: dict) -> None:
    """Tests that the digital and digital-analog qfts return the same result."""
    qc_qft_digital = QuantumCircuit(
        n_qubits, qft(n_qubits, strategy=Strategy.DIGITAL, **param_dict)
    )

    qft_analog_block = hamiltonian_factory(
        n_qubits, interaction=Interaction.NN, random_strength=True
    )

    qc_qft_digital_analog = QuantumCircuit(
        n_qubits,
        qft(n_qubits, strategy=Strategy.SDAQC, gen_build=qft_analog_block, **param_dict),
    )
    model_digital = QuantumModel(qc_qft_digital)
    model_analog = QuantumModel(qc_qft_digital_analog)

    wf_init = random_state(n_qubits)
    wf_digital = model_digital.run(values={}, state=wf_init)
    wf_analog = model_analog.run(values={}, state=wf_init)

    assert equivalent_state(wf_digital, wf_analog, atol=ATOL_64)
