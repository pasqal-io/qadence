from __future__ import annotations

import numpy as np
import pytest
import torch

import qadence as qd
from qadence import analog_feature_map, rydberg_feature_map, rydberg_hea, rydberg_tower_feature_map


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


@pytest.mark.parametrize("basis", [qd.BasisSet.FOURIER, qd.BasisSet.CHEBYSHEV])
def test_analog_feature_map(basis: qd.BasisSet) -> None:
    pname = "x"
    mname = "mult"
    fm = analog_feature_map(
        param=pname, op=qd.AnalogRY, fm_type=basis, multiplier=qd.VariationalParameter(mname)
    )
    assert isinstance(fm, qd.ConstantAnalogRotation)
    assert fm.parameters.phase == -np.pi / 2
    assert fm.parameters.delta == 0.0

    params = list(fm.parameters.alpha.free_symbols)
    assert len(params) == 2
    assert pname in params and mname in params


@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 3.0, 4.0]])
def test_rydberg_feature_map(weights: list[float] | None) -> None:
    n_qubits = 4

    fm = rydberg_feature_map(n_qubits, param="x", weights=weights)
    assert len(fm) == n_qubits
    assert all([isinstance(b, qd.ConstantAnalogRotation) for b in fm.blocks])

    circuit = qd.QuantumCircuit(n_qubits, fm)
    observable = qd.total_magnetization(n_qubits)
    model = qd.QuantumModel(circuit, observable=observable)

    values = {"x": torch.rand(1)}
    expval = model.expectation(values)
    expval.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_rydberg_tower_feature_map() -> None:
    n_qubits = 4

    fm1 = rydberg_tower_feature_map(n_qubits, param="x")
    fm2 = rydberg_feature_map(n_qubits, param="x", weights=[1.0, 2.0, 3.0, 4.0])

    for b1, b2 in zip(fm1.blocks, fm2.blocks):
        assert b1.parameters.alpha == b2.parameters.alpha  # type:ignore [attr-defined]
