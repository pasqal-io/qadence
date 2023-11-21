from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qadence.analog import IdealDevice, RydbergDevice, add_background_hamiltonian
from qadence.backend import BackendConfiguration
from qadence.logger import get_logger
from qadence.transpile import (
    blockfn_to_circfn,
    chain_single_qubit_ops,
    flatten,
    scale_primitive_blocks_only,
)
from qadence.types import AlgoHEvo

logger = get_logger(__name__)


def default_passes(config: Configuration) -> list[Callable]:
    return [
        lambda circ: add_background_hamiltonian(circ, device=config.device),
        lambda circ: blockfn_to_circfn(chain_single_qubit_ops)(circ)
        if config.use_single_qubit_composition
        else blockfn_to_circfn(flatten)(circ),
        blockfn_to_circfn(scale_primitive_blocks_only),
    ]


@dataclass
class Configuration(BackendConfiguration):
    algo_hevo: AlgoHEvo = AlgoHEvo.EXP
    """Determine which kind of Hamiltonian evolution algorithm to use."""

    n_steps_hevo: int = 100
    """Default number of steps for the Hamiltonian evolution."""

    use_gradient_checkpointing: bool = False
    """Use gradient checkpointing.

    Recommended for higher-order optimization tasks.
    """

    use_single_qubit_composition: bool = False
    """Composes chains of single qubit gates into a single matmul if possible."""

    device: RydbergDevice = IdealDevice()
    """The device including the specs for the emulated-analog interface."""

    loop_expectation: bool = False
    """When computing batches of expectation values, only allocate one wavefunction.

    Loop over the batch of parameters to only allocate a single wavefunction at any given time.
    """
