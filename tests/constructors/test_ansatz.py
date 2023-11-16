from __future__ import annotations

import pytest
from torch import Size

from qadence import (
    CNOT,
    CRX,
    RX,
    RZ,
    Interaction,
    QuantumCircuit,
    QuantumModel,
    VariationalParameter,
    Z,
    chain,
    hamiltonian_factory,
    hea,
    identity_initialized_ansatz,
    kron,
)
from qadence.blocks import AbstractBlock, has_duplicate_vparams
from qadence.types import Strategy


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("entangler", [CNOT, CRX])
def test_hea_duplicate_params(n_qubits: int, depth: int, entangler: AbstractBlock) -> None:
    """Tests that HEAs are initialized with correct parameter namings."""
    common_params = {
        "n_qubits": n_qubits,
        "depth": depth,
        "operations": [RZ, RX, RZ],
        "entangler": entangler,
    }
    hea1 = hea(n_qubits=n_qubits, depth=depth, operations=[RZ, RX, RZ], entangler=entangler)
    hea2 = hea(n_qubits=n_qubits, depth=depth, operations=[RZ, RX, RZ], entangler=entangler)
    block1 = chain(hea1, hea2)
    assert has_duplicate_vparams(block1)
    hea1 = hea(
        n_qubits=n_qubits,
        depth=depth,
        operations=[RZ, RX, RZ],
        entangler=entangler,
        param_prefix="0",
    )
    hea2 = hea(
        n_qubits=n_qubits,
        depth=depth,
        operations=[RZ, RX, RZ],
        entangler=entangler,
        param_prefix="1",
    )
    block2 = chain(hea1, hea2)
    assert not has_duplicate_vparams(block2)


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("hamiltonian", ["fixed_global", "parametric_local"])
def test_hea_sDAQC(n_qubits: int, depth: int, hamiltonian: str) -> None:
    if hamiltonian == "fixed_global":
        entangler = hamiltonian_factory(n_qubits, interaction=Interaction.NN)
    if hamiltonian == "parametric_local":
        x = VariationalParameter("x")
        entangler = x * kron(Z(0), Z(1))
    hea1 = hea(
        n_qubits=n_qubits,
        depth=depth,
        operations=[RZ, RX, RZ],
        entangler=entangler,
        strategy=Strategy.SDAQC,
    )
    # Variational parameters in the digital-analog entangler
    # are not created automatically by the hea function, but
    # by passing them in the entangler. Thus for depth larger
    # than 1 we do get duplicate vparams:
    if hamiltonian == "fixed_global":
        assert not has_duplicate_vparams(hea1)
    if hamiltonian == "parametric_local":
        assert has_duplicate_vparams(hea1)


@pytest.mark.parametrize("n_qubits", [2, 5])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("strategy", [Strategy.DIGITAL, Strategy.SDAQC])
def test_hea_forward(n_qubits: int, depth: int, strategy: Strategy) -> None:
    hea1 = hea(
        n_qubits=n_qubits,
        depth=depth,
        operations=[RZ, RX, RZ],
        strategy=strategy,
    )
    circuit = QuantumCircuit(n_qubits, hea1)
    model = QuantumModel(circuit)

    wf = model.run({})
    assert wf.shape == Size([1, 2**n_qubits])


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("entangler", [CNOT, CRX])
def test_iia_duplicate_params(n_qubits: int, depth: int, entangler: AbstractBlock) -> None:
    """Tests that IIAs are initialized with correct parameter namings."""
    iia1 = identity_initialized_ansatz(
        n_qubits=n_qubits, depth=depth, rotations=[RZ, RX, RZ], entangler=entangler
    )
    iia2 = identity_initialized_ansatz(
        n_qubits=n_qubits, depth=depth, rotations=[RZ, RX, RZ], entangler=entangler
    )
    block = chain(iia1, iia2)
    assert has_duplicate_vparams(block)


@pytest.mark.parametrize("n_qubits", [2, 5])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("entangler", [CNOT, CRX])
def test_iia_forward(n_qubits: int, depth: int, entangler: AbstractBlock) -> None:
    iia = identity_initialized_ansatz(
        n_qubits=n_qubits, depth=depth, rotations=[RZ, RX, RZ], entangler=entangler
    )
    circuit = QuantumCircuit(n_qubits, iia)
    model = QuantumModel(circuit)

    wf = model.run({})
    assert wf.shape == Size([1, 2**n_qubits])
