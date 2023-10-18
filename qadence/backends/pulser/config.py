from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pulser_simulation.simconfig import SimConfig

from qadence.backend import BackendConfiguration
from qadence.blocks.analog import Interaction

from .devices import Device


@dataclass
class Configuration(BackendConfiguration):
    device_type: Device = Device.IDEALIZED
    """The type of quantum Device to use in the simulations. Choose
    between IDEALIZED and REALISTIC. This influences pulse duration, max
    amplitude, minimum atom spacing and other properties of the system"""

    spacing: Optional[float] = None
    """Atomic spacing for Pulser register"""

    sampling_rate: float = 1.0
    """Sampling rate to be used for local simulations. Set to 1.0
    to avoid any interpolation in the solving procedure"""

    method_solv: str = "adams"
    """Solver method to pass to the Qutip solver"""

    n_steps_solv: float = 1e8
    """Number of solver steps to pass to the Qutip solver"""

    sim_config: Optional[SimConfig] = None
    """Simulation configuration with optional noise options"""

    with_modulation: bool = False
    """Add laser modulation to the local execution. This will take
    into account realistic laser pulse modulation when simulating
    a pulse sequence"""

    amplitude_local: Optional[float] = None
    """Default pulse amplitude on local channel"""

    amplitude_global: Optional[float] = None
    """Default pulse amplitude on global channel"""

    detuning: Optional[float] = None
    """Default value for the detuning pulses"""

    interaction: Interaction = Interaction.NN
    """Type of interaction introduced in the Hamiltonian. Currently, only
    NN interaction is support. XY interaction is possible but not implemented"""

    def __post_init__(self) -> None:
        if self.sim_config is not None and not isinstance(self.sim_config, SimConfig):
            raise TypeError("Wrong 'sim_config' attribute type, pass a valid SimConfig object!")
