from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Callable

from qadence.analog import add_background_hamiltonian
from qadence.backend import BackendConfiguration
from qadence.transpile import (
    blockfn_to_circfn,
    flatten,
    scale_primitive_blocks_only,
)

logger = getLogger(__name__)


def default_passes(config: Configuration) -> list[Callable]:
    passes: list = []

    # Replaces AnalogBlocks with respective HamEvo in the circuit block tree:
    passes.append(add_background_hamiltonian)

    # Flattens nested composed blocks:
    passes.append(lambda circ: blockfn_to_circfn(flatten)(circ))

    # Pushes block scales into the leaves of the block tree:
    passes.append(blockfn_to_circfn(scale_primitive_blocks_only))

    return passes


@dataclass
class Configuration(BackendConfiguration):
    n_eqs: int | None = None
    """Number of equations to use in aGPSR calculations."""

    shift_prefac: float = 0.5
    """Prefactor governing the magnitude of parameter shift values.

    Select smaller value if spectral gaps are large.
    """

    gap_step: float = 1.0
    """Step between generated pseudo-gaps when using aGPSR algorithm."""

    lb: float | None = None
    """Lower bound of optimal shift value search interval."""

    ub: float | None = None
    """Upper bound of optimal shift value search interval."""
