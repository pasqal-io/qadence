from __future__ import annotations

import numpy as np
import pytest
import torch
from pulser_simulation.simconfig import SimConfig

from qadence.backends.pulser import Backend
from qadence.blocks.utils import chain
from qadence.circuit import QuantumCircuit
from qadence.divergences import js_divergence
from qadence.operations import RY, entangle
from qadence.register import Register

SEED = 42
DEFAULT_SPACING = 8.0


def test_configuration() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    blocks = chain(entangle(892, qubit_support=(0, 1)), RY(0, torch.pi / 2))
    register = Register(2, spacing=DEFAULT_SPACING)
    circuit = QuantumCircuit(register, blocks)

    # first try the standard execution with default configuration
    backend1 = Backend()
    seq1 = backend1.circuit(circuit)
    sample1 = backend1.sample(seq1, n_shots=500)[0]

    # then add some noise and a different sampling rate
    sim_config = SimConfig(noise=("SPAM",), runs=10, eta=0.5)
    sampling_rate = 0.1

    # standard configuration method using default configuration class
    conf = Backend.default_configuration()
    conf.sim_config = sim_config
    conf.sampling_rate = sampling_rate
    backend2 = Backend(config=conf)
    seq2 = backend2.circuit(circuit)
    sample2 = backend2.sample(seq2, n_shots=500)[0]

    div = js_divergence(sample1, sample2)
    assert not np.isclose(div, 0.0, rtol=1e-2, atol=1e-2)


def test_configuration_as_dict() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    blocks = chain(entangle(892, qubit_support=(0, 1)), RY(0, torch.pi / 2))
    register = Register(2, spacing=DEFAULT_SPACING)
    circuit = QuantumCircuit(register, blocks)

    # first try the standard execution with default configuration
    backend1 = Backend()
    seq1 = backend1.circuit(circuit)
    sample1 = backend1.sample(seq1, n_shots=500)[0]

    # then add some noise and a different sampling rate
    sim_config = SimConfig(noise=("SPAM",), runs=10, eta=0.5)
    sampling_rate = 0.1

    conf = {"sim_config": sim_config, "sampling_rate": sampling_rate}
    backend2 = Backend(config=conf)  # type: ignore[arg-type]
    seq2 = backend2.circuit(circuit)
    sample2 = backend2.sample(seq2, n_shots=500)[0]

    div = js_divergence(sample1, sample2)
    assert not np.isclose(div, 0.0, rtol=1e-2, atol=1e-2)

    wrong_conf = {"wrong": "value"}
    with pytest.raises(ValueError):
        backend3 = Backend(config=wrong_conf)  # type: ignore[arg-type]
