from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qadence.backend import BackendConfiguration
from qadence.logger import get_logger
from qadence.transpile import (
    blockfn_to_circfn,
    flatten,
    scale_primitive_blocks_only,
)

logger = get_logger(__name__)


def default_passes(config: Configuration) -> list[Callable]:
    return [
        flatten,
        blockfn_to_circfn(scale_primitive_blocks_only),
    ]


@dataclass
class Configuration(BackendConfiguration):
    pass
