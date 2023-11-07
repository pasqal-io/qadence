from __future__ import annotations

import pytest

import qadence as qd
from qadence import rydberg_hea


@pytest.mark.parametrize("detunings", [True, False])
@pytest.mark.parametrize("drives", [True, False])
@pytest.mark.parametrize("phase", [True, False])
def test_rydberg_hea_construction(detunings: bool, drives: bool, phase: bool) -> None:
    n_qubits = 4
    n_layers = 2
    register = qd.Register.line(n_qubits)

    ansatz = rydberg_hea(
        register,
        n_layers=n_layers,
        addressable_detuning=detunings,
        addressable_drive=drives,
        tunable_phase=phase,
    )
    assert isinstance(ansatz, qd.CompositeBlock)
    assert len(ansatz.blocks) == n_layers

    drive_layer = ansatz.blocks[0].blocks[0]  # type:ignore [attr-defined]
    wait_layer = ansatz.blocks[0].blocks[1]  # type:ignore [attr-defined]
    det_layer = ansatz.blocks[0].blocks[2]  # type:ignore [attr-defined]

    ndrive_params = len(drive_layer.parameters.names())
    ndet_params = len(det_layer.parameters.names())

    assert ndet_params == 1 if not detunings else n_qubits + 1
    assert len(wait_layer.parameters.names()) == 1
    if not phase:
        # the +2 comes from the time evolution parameter and the scaling factor
        assert ndrive_params == 1 if not drives else n_qubits + 2
    else:
        assert ndrive_params == 4 if not drives else n_qubits + 4


def test_rydberg_hea_differentiation() -> None:
    n_qubits = 4
    n_layers = 2
    register = qd.Register.line(n_qubits)

    ansatz = rydberg_hea(
        register,
        n_layers=n_layers,
        addressable_detuning=True,
        addressable_drive=True,
        tunable_phase=True,
    )

    circuit = qd.QuantumCircuit(n_qubits, ansatz)
    observable = qd.hamiltonian_factory(register, detuning=qd.X)
    model = qd.QuantumModel(circuit, observable=observable)

    expval = model.expectation({})
    expval.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
