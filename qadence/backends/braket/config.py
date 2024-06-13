from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable

from qadence.backend import BackendConfiguration
from qadence.transpile import digitalize, fill_identities

logger = getLogger(__name__)

default_passes: list[Callable] = [fill_identities, digitalize]


@dataclass
class Configuration(BackendConfiguration):
    # FIXME: currently not used
    cloud_credentials: dict = field(default_factory=dict)
    """Credentials for connecting to the cloud.

    Execution on the QPUs available on Amazon Braket.
    """
