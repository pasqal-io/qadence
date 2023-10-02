from __future__ import annotations

from dataclasses import dataclass, field

from qadence.backend import BackendConfiguration


@dataclass
class Configuration(BackendConfiguration):
    # FIXME: currently not used
    # credentials for connecting to the cloud
    # and executing on real QPUs
    cloud_credentials: dict = field(default_factory=dict)
    # Braket requires gate-level parameters
    use_gate_params = True
