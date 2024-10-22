from __future__ import annotations

import pytest

import strategies as st  # type: ignore
from hypothesis import given, settings
import torch
from qadence import set_noise, QuantumCircuit, DigitalNoiseType, DigitalNoise
import random

list_noises = [DigitalNoiseType(noise.value) for noise in DigitalNoiseType]

@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise(protocol: str, circuit: QuantumCircuit) -> None:

    for block in circuit.block.blocks:
        assert block.noise is None
    noise = DigitalNoise(protocol, error_probability = 0.2)
    assert noise.len == 1
    set_noise(circuit, noise)

    for block in circuit.block.blocks:
        assert block.noise is not None

@pytest.mark.parametrize("protocol", list_noises)
@given(st.digital_circuits())
@settings(deadline=None)
def test_set_noise_restricted(protocol: str, circuit: QuantumCircuit) -> None:

    noise = DigitalNoise(protocol, error_probability = 0.2)
    assert noise.len == 1
    index_random_block = random.randint(0, len(circuit.block.blocks) - 1)
    type_target = type(circuit.block.blocks[index_random_block])
    set_noise(circuit, noise, target_class=type_target)

    for block in circuit.block.blocks:
        if isinstance(block, type_target):
            assert block.noise is not None
        else:
            if block.noise is not None:
                print("la merde", block, block.noise, type_target)
            assert block.noise is None