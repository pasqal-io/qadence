from __future__ import annotations

import random

import pytest
import strategies as st  # type: ignore
from hypothesis import given, settings

from qadence import DigitalNoiseType, NoiseHandler, QuantumCircuit, set_noise

list_noises = [DigitalNoiseType(noise.value) for noise in DigitalNoiseType]


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
