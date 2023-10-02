from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qadence.backend import BackendConfiguration
from qadence.types import AlgoHEvo, Interaction


@dataclass
class Configuration(BackendConfiguration):
    # FIXME: currently not used
    # determine which kind of Hamiltonian evolution
    # algorithm to use
    algo_hevo: AlgoHEvo = AlgoHEvo.EXP

    # number of steps for the Hamiltonian evolution
    n_steps_hevo: int = 100

    use_gradient_checkpointing: bool = False
    """Use gradient checkpointing. Recommended for higher-order optimization tasks."""

    use_single_qubit_composition: bool = False
    """Composes chains of single qubit gates into a single matmul if possible."""

    interaction: Callable | Interaction | str = Interaction.NN
    """Digital-analog emulation interaction that is used for `AnalogBlock`s."""

    loop_expectation: bool = False
    """When computing batches of expectation values, only allocate one wavefunction and loop over
    the batch of parameters to only allocate a single wavefunction at any given time."""
