from __future__ import annotations

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
from qadence.constructors import (
    total_magnetization,
)
from qadence.errors import Errors
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
        (0.2, total_magnetization(4), BackendName.PYQTORCH),
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
    ).sample(error=Errors(protocol=Errors.READOUT, options=None))

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
    error = Errors(protocol=Errors.READOUT, options=None).get_error_fn()
    noisy_samples = error(counters=samples, n_qubits=n_qubits, error_probability=error_probability)
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
