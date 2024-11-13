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
from qadence.backends import backend_factory
from qadence.types import BackendName, DiffMode

list_noises = [noise for noise in NoiseProtocol.DIGITAL]


def test_serialization() -> None:
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.2})
    serialized_noise = NoiseHandler._from_dict(noise._to_dict())
    assert noise == serialized_noise


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise(protocol: str, circuit: QuantumCircuit) -> None:
    all_blocks = circuit.block.blocks if hasattr(circuit.block, "blocks") else [circuit.block]
    for block in all_blocks:
        assert block.noise is None
    noise = NoiseHandler(protocol, {"error_probability": 0.2})
    assert len(noise.protocol) == 1
    set_noise(circuit, noise)

    for block in all_blocks:
        assert block.noise is not None


@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise_restricted(protocol: str, circuit: QuantumCircuit) -> None:
    noise = NoiseHandler(protocol, {"error_probability": 0.2})
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
        NoiseProtocol.DIGITAL.BITFLIP,
        [NoiseProtocol.DIGITAL.BITFLIP, NoiseProtocol.DIGITAL.PHASEFLIP],
    ],
)
def test_run_digital(noisy_config: NoiseProtocol | list[NoiseProtocol]) -> None:
    block = kron(H(0), Z(1))
    circuit = QuantumCircuit(2, block)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    noise = NoiseHandler(noisy_config, {"error_probability": 0.1})

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
        NoiseProtocol.DIGITAL.BITFLIP,
        [NoiseProtocol.DIGITAL.BITFLIP, NoiseProtocol.DIGITAL.PHASEFLIP],
    ],
)
def test_expectation_digital_noise(noisy_config: NoiseProtocol | list[NoiseProtocol]) -> None:
    block = kron(H(0), Z(1))
    circuit = QuantumCircuit(2, block)
    observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)
    noise = NoiseHandler(noisy_config, {"error_probability": 0.1})
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


@pytest.mark.parametrize(
    "noise_config",
    [
        NoiseProtocol.READOUT,
        NoiseProtocol.DIGITAL.BITFLIP,
        [NoiseProtocol.DIGITAL.BITFLIP, NoiseProtocol.DIGITAL.PHASEFLIP],
    ],
)
def test_append(noise_config: NoiseProtocol | list[NoiseProtocol]) -> None:
    options = {"error_probability": 0.1}
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)

    len_noise_config = len(noise_config) if isinstance(noise_config, list) else 1
    noise.append(NoiseHandler(noise_config, options))

    assert len(noise.protocol) == (len_noise_config + 1)


def test_equality() -> None:
    options = {"error_probability": 0.1}
    noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)
    noise.append(NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options))

    noise2 = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, options)
    noise2.bitflip(options)

    assert noise == noise2
