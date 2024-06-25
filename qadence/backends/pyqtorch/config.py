from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Callable

from pyqtorch.utils import SolverType

from qadence.analog import add_background_hamiltonian
from qadence.backend import BackendConfiguration
from qadence.transpile import (
    blockfn_to_circfn,
    chain_single_qubit_ops,
    flatten,
    scale_primitive_blocks_only,
)
from qadence.types import AlgoHEvo

logger = getLogger(__name__)


def default_passes(config: Configuration) -> list[Callable]:
    passes: list = []

    # Replaces AnalogBlocks with respective HamEvo in the circuit block tree:
    passes.append(add_background_hamiltonian)

    if config.use_single_qubit_composition:
        # Composes chains of single-qubit gates into a single unitary before applying to the state:
        passes.append(lambda circ: blockfn_to_circfn(chain_single_qubit_ops)(circ))
    else:
        # Flattens nested composed blocks:
        passes.append(lambda circ: blockfn_to_circfn(flatten)(circ))

    # Pushes block scales into the leaves of the block tree:
    passes.append(blockfn_to_circfn(scale_primitive_blocks_only))

    return passes


@dataclass
class Configuration(BackendConfiguration):
    algo_hevo: AlgoHEvo = AlgoHEvo.EXP
    """Determine which kind of Hamiltonian evolution algorithm to use."""

    ode_solver: SolverType = SolverType.DP5_SE
    """Determine which ODE solver to use for time-dependent blocks."""

    n_steps_hevo: int = 100
    """Default number of steps for the Hamiltonian evolution."""

    use_gradient_checkpointing: bool = False
    """Use gradient checkpointing.

    Recommended for higher-order optimization tasks.
    """

    use_single_qubit_composition: bool = False
    """Composes chains of single qubit gates into a single matmul if possible."""

    loop_expectation: bool = False
    """When computing batches of expectation values, only allocate one wavefunction.

    Loop over the batch of parameters to only allocate a single wavefunction at any given time.
    """
