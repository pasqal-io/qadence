from __future__ import annotations

import pytest
import torch
from sympy import acos
import qadence as qd
from qadence.operations import *
from qadence import BackendName
from qadence.errors import Errors


@pytest.mark.parametrize("backend", [BackendName.BRAKET, BackendName.PYQTORCH, BackendName.PULSER])
def test_readout_error(backend: BackendName) -> None:
    n_qubits = 5
    fidelity = 0.1
    fp = qd.FeatureParameter("phi")
    feature_map = qd.kron(RX(i, 2 * acos(fp)) for i in range(n_qubits))
    inputs = {"phi": torch.rand(1)}
    # sample
    samples = qd.sample(feature_map, n_shots=1000, values=inputs, backend=backend, error=None)
    # introduce errors
    error = Errors(protocol=Errors.READOUT, options=None).get_error_fn()
    noisy_samples = error(counters=samples, n_qubits=n_qubits, fidelity=fidelity)
    # compare that the results are with an error of 10% (the default fidelity)
    assert all(
        [
            True
            if (
                samples[0]["bitstring"] < int(count + count * fidelity)
                or samples[0]["bitstring"] > int(count - count * fidelity)
            )
            else False
            for bitstring, count in noisy_samples[0].items()
        ]
    )
