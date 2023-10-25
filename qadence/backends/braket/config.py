from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from qadence.backend import BackendConfiguration
from qadence.logger import get_logger
from qadence.transpile import digitalize, fill_identities

logger = get_logger(__name__)

default_passes: list[Callable] = [fill_identities, digitalize]


@dataclass
class Configuration(BackendConfiguration):
    # FIXME: currently not used
    cloud_credentials: dict = field(default_factory=dict)
    """Credentials for connecting to the cloud
    and executing on the QPUs available on Amazon Braket"""

    # # this post init is needed because of the way dataclasses
    # # inherit attributes and class MRO. See here:
    # # https://stackoverflow.com/a/53085935
    # def __post_init__(self) -> None:
    #     super().__post_init__()
    #     if self.transpilation_passes is None:
    #         # by default make sure that we don't have empty wires
    #         # in case no custom transpilation passes are sent
    #         self.transpilation_passes = [fill_identities, digitalize]
