from __future__ import annotations

from collections import Counter
from itertools import chain

import pytest
import strategies as st
import torch
from hypothesis import given, settings
from sympy import acos
from torch import Tensor

import qadence as qd
from qadence import BackendName
from qadence.blocks import (
    AbstractBlock,
    add,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.errors import Errors
from qadence.errors.readout import bs_corruption
from qadence.measurements.protocols import Measurements
from qadence.models import QuantumModel
from qadence.operations import (
    CNOT,
    RX,
    RZ,
    HamEvo,
    X,
    Y,
    Z,
)
from qadence.types import DiffMode


@pytest.mark.parametrize(
    "error_probability, counters, exp_corrupted_counters, n_qubits",
    [
        (
            1.0,
            [Counter({"00": 27, "01": 23, "10": 24, "11": 26})],
            [Counter({"11": 27, "10": 23, "01": 24, "00": 26})],
            2,
        ),
        (
            1.0,
            [Counter({"001": 27, "010": 23, "101": 24, "110": 26})],
            [Counter({"110": 27, "101": 23, "010": 24, "001": 26})],
            3,
        ),
    ],
)
def test_bitstring_corruption(
    error_probability: float, counters: list, exp_corrupted_counters: list, n_qubits: int
) -> None:
    corrupted_bitstrings = [
        bs_corruption(
            bitstring=bitstring,
            n_shots=n_shots,
            error_probability=error_probability,
            n_qubits=n_qubits,
        )
        for bitstring, n_shots in counters[0].items()
    ]
    corrupted_counters = [Counter(chain(*corrupted_bitstrings))]
    breakpoint()
    assert corrupted_counters == exp_corrupted_counters


@pytest.mark.parametrize(
    "error_probability, block, backend",
    [
        (0.1, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.15, add(Z(0), Z(1), Z(2)), BackendName.BRAKET),
        (0.01, kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), BackendName.BRAKET),
        (0.1, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
        (0.1, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.BRAKET),
        (
            0.05,
            kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
            + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
            BackendName.PULSER,
        ),
        (0.001, HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05), BackendName.BRAKET),
        (0.12, HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001), BackendName.BRAKET),
        (
            0.1,
            HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
            BackendName.BRAKET,
        ),
        (0.1, kron(X(0), X(1)), BackendName.PYQTORCH),
        (0.01, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.PYQTORCH),
        (0.01, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH),
        (
            0.1,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
            BackendName.PYQTORCH,
        ),
        (0.1, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.PYQTORCH),
        (0.05, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.PYQTORCH),
        (0.2, hamiltonian_factory(4, detuning=Z), BackendName.PYQTORCH),
        (0.1, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.PYQTORCH),
    ],
)
def test_readout_error_quantum_model(
    error_probability: float, block: AbstractBlock, backend: BackendName
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"

    err_free = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample()

    noisy = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample(error=Errors(protocol=Errors.READOUT))

    assert len(noisy[0]) <= 2 ** block.n_qubits and len(noisy[0]) > len(err_free[0])
    assert all(
        [
            True
            if (
                err_free[0]["bitstring"] < int(count + count * error_probability)
                or err_free[0]["bitstring"] > int(count - count * error_probability)
            )
            else False
            for bitstring, count in noisy[0].items()
        ]
    )


@pytest.mark.parametrize("backend", [BackendName.BRAKET, BackendName.PYQTORCH, BackendName.PULSER])
def test_readout_error_backends(backend: BackendName) -> None:
    n_qubits = 5
    error_probability = 0.1
    fp = qd.FeatureParameter("phi")
    feature_map = qd.kron(RX(i, 2 * acos(fp)) for i in range(n_qubits))
    inputs = {"phi": torch.rand(1)}
    # sample
    samples = qd.sample(feature_map, n_shots=1000, values=inputs, backend=backend, error=None)
    # introduce errors
    options = {"error_probability": error_probability}
    error = Errors(protocol=Errors.READOUT, options=options).get_error_fn()
    noisy_samples = error(counters=samples, n_qubits=n_qubits)
    # compare that the results are with an error of 10% (the default error_probability)
    assert all(
        [
            True
            if (
                samples[0]["bitstring"] < int(count + count * error_probability)
                or samples[0]["bitstring"] > int(count - count * error_probability)
            )
            else False
            for bitstring, count in noisy_samples[0].items()
        ]
    )


@pytest.mark.parametrize("measurement_proto", [Measurements.TOMOGRAPHY, Measurements.SHADOW])
@given(st.restricted_batched_circuits())
@settings(deadline=None)
def test_readout_error_with_measurements(
    measurement_proto: Measurements, circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]
) -> None:
    circuit, inputs = circ_and_vals
    # print(circuit, inputs)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    model = QuantumModel(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    # model.backend.backend.config._use_gate_params = True

    error = Errors(protocol=Errors.READOUT)
    measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options={"n_shots": 1000})

    measured = model.expectation(values=inputs, measurement=measurement)
    noisy = model.expectation(values=inputs, measurement=measurement, error=error)
    exact = model.expectation(values=inputs)
    # breakpoint()
    if exact.numel() > 1:
        exact_values = torch.abs(exact)
        for noisy_value, exact_value in zip(noisy, exact):
            noisy_val = noisy_value.item()
            exact_val = exact_value.item()
            atol = exact_val / 3.0 if exact_val != 0.0 else 0.33
            assert torch.allclose(noisy_value, exact_value, atol=atol)

    else:
        exact_value = torch.abs(exact).item()
        # print(f"exact {exact_value}")
        atol = exact_value / 3.0 if exact_value != 0.0 else 0.33
        assert torch.allclose(noisy, exact, atol=atol)
