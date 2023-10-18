from __future__ import annotations

from dataclasses import dataclass, field

from qadence.backend import BackendConfiguration
from qadence.transpile import digitalize, fill_identities


@dataclass
class Configuration(BackendConfiguration):
    # FIXME: currently not used
    # credentials for connecting to the cloud
    # and executing on real QPUs
    cloud_credentials: dict = field(default_factory=dict)

    # this post init is needed because of the way dataclasses
    # inherit attributes and class MRO. See here:
    # https://stackoverflow.com/a/53085935
    def __post_init__(self):
        # make sure that we don't have empty wires in case no
        # custom transpilation passes are sent
        if len(self.transpilation_passes) == 0:
            self.transpilation_passes = [fill_identities, digitalize]

        # Braket requires gate-level parameters
        self.use_gate_params = True
