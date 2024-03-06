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
    pass
