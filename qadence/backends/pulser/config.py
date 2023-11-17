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
    device_type: Device = Device.IDEALIZED
    """The type of quantum Device to use in the simulations.

    Choose
    between IDEALIZED and REALISTIC. This influences pulse duration, max
    amplitude, minimum atom spacing and other properties of the system
    """

    sampling_rate: float = 1.0
    """Sampling rate to be used for local simulations.

    Set to 1.0
    to avoid any interpolation in the solving procedure
    """

    spacing: Optional[float] = None
    """Spacing that multiplies the coordinates of the register.

    FIXME: This is deprecated, spacing is now controlled in the Register,
    as detailed in the register tutorial.
    """

    method_solv: str = "adams"
    """Solver method to pass to the Qutip solver."""

    n_steps_solv: float = 1e8
    """Number of solver steps to pass to the Qutip solver."""

    sim_config: Optional[SimConfig] = None
    """Simulation configuration with optional noise options."""

    with_modulation: bool = False
    """Add laser modulation to the local execution.

    This will take
    into account realistic laser pulse modulation when simulating
    a pulse sequence
    """

    amplitude_local: Optional[float] = None
    """Default pulse amplitude on local channel."""

    amplitude_global: Optional[float] = None
    """Default pulse amplitude on global channel."""

    detuning: Optional[float] = None
    """Default value for the detuning pulses."""

    interaction: Interaction = Interaction.NN
    """Type of interaction introduced in the Hamiltonian.

    Currently, only
    NN interaction is support. XY interaction is possible but not implemented
    """

    # configuration for cloud simulations
    cloud_configuration: Optional[CloudConfiguration] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sim_config is not None and not isinstance(self.sim_config, SimConfig):
            raise TypeError("Wrong 'sim_config' attribute type, pass a valid SimConfig object!")

        if isinstance(self.cloud_configuration, dict):
            self.cloud_configuration = CloudConfiguration(**self.cloud_configuration)
