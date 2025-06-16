from __future__ import annotations

import random
from functools import reduce
from operator import add

import pytest
import strategies as st  # type: ignore
import torch
from hypothesis import given, settings

from qadence import (
    H,
    QuantumCircuit,
    QuantumModel,
    Z,
    hamiltonian_factory,
    kron,
    set_noise,
)
from qadence.noise import NoiseCategory, available_protocols
from qadence.backends import backend_factory
from qadence.types import BackendName, DiffMode

list_noises = [noise for noise in NoiseCategory.DIGITAL]


def test_serialization() -> None:
    noise = available_protocols.Bitflip(error_definition=0.2)
    serialized_noise = available_protocols.Bitflip(noise.model_dump())
    assert noise == serialized_noise


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise(protocol: str, circuit: QuantumCircuit) -> None:
    all_blocks = circuit.block.blocks if hasattr(circuit.block, "blocks") else [circuit.block]
    for block in all_blocks:
        assert block.noise is None
    noise = available_protocols.PrimitiveNoise(protocol=protocol, error_definition=0.2)
    assert len(noise.protocol) == 1
    set_noise(circuit, noise)

    for block in all_blocks:
        assert block.noise is not None


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise_restricted(protocol: str, circuit: QuantumCircuit) -> None:
    noise = available_protocols.PrimitiveNoise(protocol=protocol, error_definition=0.2)
    assert len(noise.protocol) == 1
    all_blocks = circuit.block.blocks if hasattr(circuit.block, "blocks") else [circuit.block]
    index_random_block = random.randint(0, len(all_blocks) - 1)
    type_target = type(all_blocks[index_random_block])
    set_noise(circuit, noise, target_class=type_target)  # type: ignore[arg-type]

    for block in all_blocks:
        if isinstance(block, type_target):
            assert block.noise is not None
        else:
            assert block.noise is None


@pytest.mark.parametrize(
    "noisy_config",
    [
        [
            NoiseCategory.DIGITAL.BITFLIP,
        ],
        [NoiseCategory.DIGITAL.BITFLIP, NoiseCategory.DIGITAL.PHASEFLIP],
    ],
)
def test_run_digital(noisy_config: list[NoiseCategory]) -> None:
    block = kron(H(0), Z(1))
    circuit = QuantumCircuit(2, block)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    noise = reduce(
        add,
        [
            available_protocols.PrimitiveNoise(protocol=protocol, error_definition=0.1)
            for protocol in noisy_config
        ],
    )

    # Construct a quantum model.
    model = QuantumModel(circuit=circuit, observable=observable)
    noiseless_output = model.run()

    set_noise(circuit, noise)
    noisy_model = QuantumModel(circuit=circuit, observable=observable)
    noisy_output = noisy_model.run()
    assert not torch.allclose(noiseless_output, noisy_output)

    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    (pyqtorch_circ, _, embed, params) = backend.convert(circuit)
    native_output = backend.run(pyqtorch_circ, embed(params, {}))

    assert torch.allclose(noisy_output, native_output)


@pytest.mark.parametrize(
    "noisy_config",
    [
        [
            NoiseCategory.DIGITAL.BITFLIP,
        ],
        [NoiseCategory.DIGITAL.BITFLIP, NoiseCategory.DIGITAL.PHASEFLIP],
    ],
)
def test_expectation_digital_noise(noisy_config: list[NoiseCategory]) -> None:
    block = kron(H(0), Z(1))
    circuit = QuantumCircuit(2, block)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    noise = reduce(
        add,
        [
            available_protocols.PrimitiveNoise(protocol=protocol, error_definition=0.1)
            for protocol in noisy_config
        ],
    )
    backend = backend_factory(backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

    # Construct a quantum model.
    model = QuantumModel(circuit=circuit, observable=observable)
    noiseless_expectation = model.expectation(values={})

    (pyqtorch_circ, pyqtorch_obs, embed, params) = backend.convert(circuit, observable)
    native_noisy_expectation = backend.expectation(
        pyqtorch_circ, pyqtorch_obs, embed(params, {}), noise=noise
    )
    assert not torch.allclose(noiseless_expectation, native_noisy_expectation)

    noisy_model = QuantumModel(circuit=circuit, observable=observable, noise=noise)
    noisy_model_expectation = noisy_model.expectation(values={})
    assert torch.allclose(noisy_model_expectation, native_noisy_expectation)

    (pyqtorch_circ, pyqtorch_obs, embed, params) = backend.convert(circuit, observable)
    noisy_converted_model_expectation = backend.expectation(
        pyqtorch_circ, pyqtorch_obs, embed(params, {})
    )

    assert torch.allclose(noisy_converted_model_expectation, native_noisy_expectation)
