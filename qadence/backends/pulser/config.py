from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pasqal_cloud import TokenProvider
from pasqal_cloud.device import EmulatorType
from pulser_simulation.simconfig import SimConfig

from qadence.backend import BackendConfiguration
from qadence.blocks.analog import Interaction

from .devices import Device

DEFAULT_CLOUD_ENV = "prod"


@dataclass
class CloudConfiguration:
    platform: EmulatorType = EmulatorType.EMU_FREE
    username: str | None = None
    password: str | None = None
    project_id: str | None = None
    environment: str = "prod"
    token_provider: TokenProvider | None = None


@dataclass
class Configuration(BackendConfiguration):
    # device type
    device_type: Device = Device.IDEALIZED

    # atomic spacing
    spacing: Optional[float] = None

    # sampling rate to be used for local simulations
    sampling_rate: float = 1.0

    # solver method to pass to the Qutip solver
    method_solv: str = "adams"

    # number of solver steps to pass to the Qutip solver
    n_steps_solv: float = 1e8

    # simulation configuration with optional noise options
    sim_config: Optional[SimConfig] = None

    # add modulation to the local execution
    with_modulation: bool = False

    # Use gate-level parameters
    use_gate_params = True

    # pulse amplitude on local channel
    amplitude_local: Optional[float] = None

    # pulse amplitude on global channel
    amplitude_global: Optional[float] = None

    # detuning value
    detuning: Optional[float] = None

    # interaction type
    interaction: Interaction = Interaction.NN

    # configuration for cloud simulations
    cloud_configuration: Optional[CloudConfiguration] = None

    def __post_init__(self) -> None:
        if self.sim_config is not None and not isinstance(self.sim_config, SimConfig):
            raise TypeError("Wrong 'sim_config' attribute type, pass a valid SimConfig object!")

        if isinstance(self.cloud_configuration, dict):
            self.cloud_configuration = CloudConfiguration(**self.cloud_configuration)
