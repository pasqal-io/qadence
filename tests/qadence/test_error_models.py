from __future__ import annotations

from collections import Counter
from itertools import chain

import pytest
import torch
from sympy import acos

import qadence as qd
from qadence import BackendName
from qadence.blocks import (
    AbstractBlock,
    add,
    kron,
)
from qadence.circuit import QuantumCircuit
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.divergences import js_divergence
from qadence.errors import Errors
from qadence.errors.readout import bs_corruption
from qadence.measurements.protocols import Measurements
from qadence.models import QuantumModel
from qadence.operations import (
    CNOT,
    RX,
    RZ,
    H,
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
    # corrupted_counters = corrupt(
    #     bitflip_proba=error_probability,
    #     counters=counters,
    #     n_qubits=n_qubits,
    # )
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
    # assert corrupted_counters[0].total() == 100 # python3.9 complains about .total in Counter
    assert sum(corrupted_counters[0].values()) == 100
    assert corrupted_counters == exp_corrupted_counters
    assert torch.allclose(
        torch.tensor(1.0 - js_divergence(corrupted_counters[0], counters[0])),
        torch.ones(1),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, 1000, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.15, 1000, add(Z(0), Z(1), Z(2)), BackendName.BRAKET),
        (0.1, 5000, kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), BackendName.BRAKET),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, 2000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
        (0.1, 1300, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.BRAKET),
        (
            0.05,
            1500,
            kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
            + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
            BackendName.PULSER,
        ),
        (0.001, 5000, HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05), BackendName.BRAKET),
        (0.12, 2000, HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001), BackendName.BRAKET),
        (
            0.1,
            1000,
            HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
            BackendName.BRAKET,
        ),
        (0.1, 100, kron(X(0), X(1)), BackendName.PYQTORCH),
        (0.1, 200, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.PYQTORCH),
        (0.01, 1000, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH),
        (
            0.1,
            2000,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
            BackendName.PYQTORCH,
        ),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.PYQTORCH),
        (0.05, 10000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.PYQTORCH),
        (0.2, 1000, hamiltonian_factory(4, detuning=Z), BackendName.PYQTORCH),
        (0.1, 500, kron(Z(0), Z(1)) + CNOT(0, 1), BackendName.PYQTORCH),
    ],
)
def test_readout_error_quantum_model(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"

    noiseless_samples: list[Counter] = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample(n_shots=n_shots)

    noisy_samples: list[Counter] = QuantumModel(
        QuantumCircuit(block.n_qubits, block), backend=backend, diff_mode=diff_mode
    ).sample(error=Errors(protocol=Errors.READOUT), n_shots=n_shots)

    # breakpoint()
    for noiseless, noisy in zip(noiseless_samples, noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        # assert noiseless.total() == noisy.total() # python3.9 complains about .total in Counter
        assert js_divergence(noiseless, noisy) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(noiseless, noisy)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )
        # print(js_divergence(noiseless, noisy))
    # assert len(noisy[0]) <= 2 ** block.n_qubits and len(noisy[0]) > len(err_free[0])
    # assert all(
    #     [
    #         True
    #         if (
    #             err_free[0]["bitstring"] < int(count + count * error_probability)
    #             or err_free[0]["bitstring"] > int(count - count * error_probability)
    #         )
    #         else False
    #         for bitstring, count in noisy[0].items()
    #     ]
    # )


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
    # breakpoint()
    # compare that the results are with an error of 10% (the default error_probability)
    for sample, noisy_sample in zip(samples, noisy_samples):
        assert sum(sample.values()) == sum(noisy_sample.values())
        # python3.9 complains about .total in Counter
        # assert sample.total() == noisy_sample.total()
        assert js_divergence(sample, noisy_sample) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(sample, noisy_sample)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )
    # assert all(
    #     [
    #         True
    #         if (
    #             samples[0]["bitstring"] < int(count + count * error_probability)
    #             or samples[0]["bitstring"] > int(count - count * error_probability)
    #         )
    #         else False
    #         for bitstring, count in noisy_samples[0].items()
    #     ]
    # )


# @pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "measurement_proto, options",
    [
        (Measurements.TOMOGRAPHY, {"n_shots": 10000}),
        (Measurements.SHADOW, {"accuracy": 0.1, "confidence": 0.1}),
    ],
)
# @given(st.restricted_batched_circuits())
# @settings(deadline=None)
def test_readout_error_with_measurements(
    measurement_proto: Measurements,
    options: dict,
    # circ_and_vals: tuple[QuantumCircuit, dict[str, Tensor]]
) -> None:
    # circuit, inputs = circ_and_vals
    circuit = QuantumCircuit(2, kron(H(0), Z(1)))
    inputs: dict = dict()
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

    model = QuantumModel(circuit=circuit, observable=observable, diff_mode=DiffMode.GPSR)
    # model.backend.backend.config._use_gate_params = True

    error = Errors(protocol=Errors.READOUT)
    measurement = Measurements(protocol=str(measurement_proto), options=options)

    # measured = model.expectation(values=inputs, measurement=measurement)
    noisy = model.expectation(values=inputs, measurement=measurement, error=error)
    exact = model.expectation(values=inputs)
    if exact.numel() > 1:
        for noisy_value, exact_value in zip(noisy, exact):
            exact_val = torch.abs(exact_value).item()
            atol = exact_val / 3.0 if exact_val != 0.0 else 0.33
            assert torch.allclose(noisy_value, exact_value, atol=atol)

    else:
        exact_value = torch.abs(exact).item()
        atol = exact_value / 3.0 if exact_value != 0.0 else 0.33
        assert torch.allclose(noisy, exact, atol=atol)