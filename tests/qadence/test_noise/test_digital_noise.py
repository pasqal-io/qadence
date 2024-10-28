from __future__ import annotations

import random

import pytest
import strategies as st  # type: ignore
import torch
from hypothesis import given, settings

from qadence import (
    H,
    NoiseHandler,
    NoiseProtocol,
    QuantumCircuit,
    QuantumModel,
    Z,
    hamiltonian_factory,
    kron,
    set_noise,
)

list_noises = [noise for noise in NoiseProtocol.DIGITAL]


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise(protocol: str, circuit: QuantumCircuit) -> None:
    all_blocks = circuit.block.blocks if hasattr(circuit.block, "blocks") else [circuit.block]
    for block in all_blocks:
        assert block.noise is None
    noise = NoiseHandler(protocol, {"error_probability": 0.2})
    assert len(noise.noise_sources) == 1
    set_noise(circuit, noise)

    for block in all_blocks:
        assert block.noise is not None


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise_restricted(protocol: str, circuit: QuantumCircuit) -> None:
    noise = NoiseHandler(protocol, {"error_probability": 0.2})
    assert len(noise.noise_sources) == 1
    all_blocks = circuit.block.blocks if hasattr(circuit.block, "blocks") else [circuit.block]
    index_random_block = random.randint(0, len(all_blocks) - 1)
    type_target = type(all_blocks[index_random_block])
    set_noise(circuit, noise, target_class=type_target)  # type: ignore[arg-type]

    for block in all_blocks:
        if isinstance(block, type_target):
            assert block.noise is not None
        else:
            assert block.noise is None


def test_run_digital() -> None:
    block = kron(H(0), Z(1))
    circuit = QuantumCircuit(2, block)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.1})

    # Construct a quantum model.
    model = QuantumModel(circuit=circuit, observable=observable)
    noiseless_exp = model.expectation()

    set_noise(circuit, noise)
    noisy_model = QuantumModel(circuit=circuit, observable=observable)
    noisy_expectation = noisy_model.expectation()
    assert not torch.allclose(noiseless_exp, noisy_expectation)
