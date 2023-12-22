from __future__ import annotations

from collections import Counter

import pytest
import torch
from metrics import MIDDLE_ACCEPTANCE
from torch import Tensor

from qadence import (
    AbstractBlock,
    AnalogRX,
    AnalogRZ,
    Mitigations,
    QuantumCircuit,
    QuantumModel,
    chain,
    entangle,
    hamiltonian_factory,
)
from qadence.divergences import js_divergence
from qadence.noise.protocols import Noise
from qadence.operations import CNOT, RX, RY, RZ, HamEvo, X, Y, Z, add, kron
from qadence.types import BackendName, DiffMode, ReadOutOptimization

pi = torch.pi


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend, optimization_type",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET, ReadOutOptimization.MLE),
        (
            0.1,
            1000,
            kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (0.15, 1000, add(Z(0), Z(1), Z(2)), BackendName.BRAKET, ReadOutOptimization.CONSTRAINED),
        (
            0.1,
            5000,
            kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.1,
            500,
            add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            2000,
            add(kron(Z(0), Z(1)), kron(X(2), X(3))),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            1300,
            kron(Z(0), Z(1)) + CNOT(0, 1),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.05,
            1500,
            kron(RZ(0, parameter=0.01), RZ(1, parameter=0.01))
            + kron(RX(0, parameter=0.01), RX(1, parameter=0.01)),
            BackendName.PULSER,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.001,
            5000,
            HamEvo(generator=kron(Z(0), Z(1)), parameter=0.05),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.12,
            2000,
            HamEvo(generator=kron(Z(0), Z(1), Z(2)), parameter=0.001),
            BackendName.BRAKET,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            1000,
            HamEvo(generator=kron(Z(0), Z(1)) + kron(Z(0), Z(1), Z(2)), parameter=0.005),
            BackendName.BRAKET,
            ReadOutOptimization.CONSTRAINED,
        ),
        (0.1, 100, kron(X(0), X(1)), BackendName.PYQTORCH, ReadOutOptimization.CONSTRAINED),
        (
            0.1,
            200,
            kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (0.01, 1000, add(Z(0), Z(1), Z(2)), BackendName.PYQTORCH, ReadOutOptimization.MLE),
        (
            0.1,
            2000,
            HamEvo(
                generator=kron(X(0), X(1)) + kron(Z(0), Z(1)) + kron(X(2), X(3)), parameter=0.005
            ),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.1,
            500,
            add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
        (
            0.05,
            10000,
            add(kron(Z(0), Z(1)), kron(X(2), X(3))),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (
            0.2,
            1000,
            hamiltonian_factory(4, detuning=Z),
            BackendName.PYQTORCH,
            ReadOutOptimization.MLE,
        ),
        (
            0.1,
            500,
            kron(Z(0), Z(1)) + CNOT(0, 1),
            BackendName.PYQTORCH,
            ReadOutOptimization.CONSTRAINED,
        ),
    ],
)
def test_readout_mitigation_quantum_model(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
    optimization_type: str,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"
    circuit = QuantumCircuit(block.n_qubits, block)
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noise = Noise(protocol=Noise.READOUT)
    mitigation = Mitigations(
        protocol=Mitigations.READOUT, options={"optimization_type": optimization_type}
    )
    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)
    noisy_samples: list[Counter] = model.sample(noise=noise, n_shots=n_shots)
    mitigated_samples: list[Counter] = model.sample(
        noise=noise, mitigation=mitigation, n_shots=n_shots
    )

    js_mitigated = js_divergence(mitigated_samples[0], noiseless_samples[0])
    js_noisy = js_divergence(noisy_samples[0], noiseless_samples[0])
    assert js_mitigated < js_noisy


@pytest.mark.parametrize(
    "error_probability, n_shots, block, backend",
    [
        (0.1, 100, kron(X(0), X(1)), BackendName.BRAKET),
        (0.1, 1000, kron(Z(0), Z(1), Z(2)) + kron(X(0), Y(1), Z(2)), BackendName.BRAKET),
        (0.1, 500, add(Z(0), Z(1), kron(X(2), X(3))) + add(X(2), X(3)), BackendName.BRAKET),
        (0.1, 2000, add(kron(Z(0), Z(1)), kron(X(2), X(3))), BackendName.BRAKET),
    ],
)
def test_compare_readout_methods(
    error_probability: float,
    n_shots: int,
    block: AbstractBlock,
    backend: BackendName,
) -> None:
    diff_mode = "ad" if backend == BackendName.PYQTORCH else "gpsr"
    circuit = QuantumCircuit(block.n_qubits, block)
    model = QuantumModel(circuit=circuit, backend=backend, diff_mode=diff_mode)

    noise = Noise(protocol=Noise.READOUT)

    noiseless_samples: list[Counter] = model.sample(n_shots=n_shots)

    mitigation_mle = Mitigations(
        protocol=Mitigations.READOUT, options={"optimization_type": ReadOutOptimization.CONSTRAINED}
    )
    mitigated_samples_mle: list[Counter] = model.sample(
        noise=noise, mitigation=mitigation_mle, n_shots=n_shots
    )

    mitigation_constrained_opt = Mitigations(
        protocol=Mitigations.READOUT, options={"optimization_type": ReadOutOptimization.MLE}
    )
    mitigated_samples_constrained_opt: list[Counter] = model.sample(
        noise=noise, mitigation=mitigation_constrained_opt, n_shots=n_shots
    )
    js_mitigated_mle = js_divergence(mitigated_samples_mle[0], noiseless_samples[0])
    js_mitigated_constrained_opt = js_divergence(
        mitigated_samples_constrained_opt[0], noiseless_samples[0]
    )
    assert abs(js_mitigated_constrained_opt - js_mitigated_mle) <= MIDDLE_ACCEPTANCE


@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type",
    [
        (
            chain(AnalogRX(pi / 2.0), AnalogRZ(pi)),
            [Z(0) + Z(1)],
            torch.linspace(0.1, 0.5, 8),
            Noise.DEPOLARIZING,
        ),
        (
            # Hardcoded time and angle for Bell state preparation.
            chain(
                entangle(383, qubit_support=(0, 1)),
                RY(0, 3.0 * pi / 2.0),
            ),
            [hamiltonian_factory(2, detuning=Z)],
            torch.linspace(0.1, 0.5, 8),
            Noise.DEPHASING,
        ),
    ],
)
def test_analog_zne_with_noise_levels(
    analog_block: AbstractBlock, observable: AbstractBlock, noise_probs: Tensor, noise_type: str
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = Noise(protocol=noise_type, options=options)
    mitigation = Mitigations(protocol=Mitigations.ANALOG_ZNE)
    exact_expectation = model.expectation()
    mitigated_expectation = model.expectation(noise=noise, mitigation=mitigation)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=1.0e-2)


# FIXME: Consider a stretchable replacement for entangle.
@pytest.mark.parametrize(
    "analog_block, observable, noise_probs, noise_type, param_values",
    [
        (
            chain(AnalogRX(pi / 2.0), AnalogRZ(pi)),
            [Z(0) + Z(1)],
            torch.tensor([0.1]),
            Noise.DEPOLARIZING,
            {},
        ),
        # (
        #     # Parameter time and harcoded angle for Bell state preparation.
        #     chain(
        #         entangle("t", qubit_support=(0, 1)),
        #         RY(0, 3.0 * pi / 2.0),
        #     ),
        #     [hamiltonian_factory(2, detuning=Z)],
        #     torch.tensor([0.1]),
        #     Noise.DEPHASING,
        #     {"t": torch.tensor([1.0])},
        # ),
    ],
)
def test_analog_zne_with_pulse_stretching(
    analog_block: AbstractBlock,
    observable: AbstractBlock,
    noise_probs: Tensor,
    noise_type: str,
    param_values: dict,
) -> None:
    circuit = QuantumCircuit(2, analog_block)
    model = QuantumModel(
        circuit=circuit, observable=observable, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR
    )
    options = {"noise_probs": noise_probs}
    noise = Noise(protocol=noise_type, options=options)
    options = {"stretches": torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])}
    mitigation = Mitigations(protocol=Mitigations.ANALOG_ZNE, options=options)
    mitigated_expectation = model.expectation(
        values=param_values, noise=noise, mitigation=mitigation
    )
    exact_expectation = model.expectation(values=param_values)
    assert torch.allclose(mitigated_expectation, exact_expectation, atol=2.0e-1)
