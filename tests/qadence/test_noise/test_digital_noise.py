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
    NoiseSource,
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
    noise = NoiseHandler.bitflip({"error_probability": 0.2})
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

    if isinstance(noise_config, list):
        noise.append(NoiseHandler(noise_config, options))
        len_noise2 = len(noise_config)
    else:
        noise.append(NoiseSource(noise_config, options))
        len_noise2 = 1

    assert len(noise.noise_sources) == (len_noise2 + 1)
