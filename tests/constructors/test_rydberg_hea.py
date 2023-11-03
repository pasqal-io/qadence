from __future__ import annotations

import pytest
from qadence_extensions.rydberg_hea import rydberg_hea

import qadence as qd


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
    assert isinstance(ansatz, qd.AbstractBlock)
    assert len(ansatz.blocks) == n_layers

    drive_layer = ansatz.blocks[0].blocks[0]
    wait_layer = ansatz.blocks[0].blocks[1]
    det_layer = ansatz.blocks[0].blocks[2]

    ndrive_params = len(drive_layer.parameters.names())
    ndet_params = len(det_layer.parameters.names())

    assert ndet_params == 1 if not detunings else n_qubits + 1
    assert len(wait_layer.parameters.names()) == 1
    if not phase:
        # the +2 comes from the time evolution parameter and the scaling factor
        assert ndrive_params == 1 if not drives else n_qubits + 2
    else:
        assert ndrive_params == 4 if not drives else n_qubits + 4


def test_rydberg_hea_training() -> None:
    pass
