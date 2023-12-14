# flake8: noqa

from .feature_maps import (
    feature_map,
    chebyshev_feature_map,
    fourier_feature_map,
    tower_feature_map,
    exp_fourier_feature_map,
)

from .ansatze import hea, build_qnn

from .iia import identity_initialized_ansatz

from .daqc import daqc_transform

from .hamiltonians import (
    hamiltonian_factory,
    ising_hamiltonian,
    single_z,
    total_magnetization,
    zz_hamiltonian,
)

from .rydberg_hea import rydberg_hea, rydberg_hea_layer
from .rydberg_feature_maps import rydberg_feature_map, analog_feature_map, rydberg_tower_feature_map

from .qft import qft

# Modules to be automatically added to the qadence namespace
__all__ = [
    "feature_map",
    "chebyshev_feature_map",
    "fourier_feature_map",
    "tower_feature_map",
    "exp_fourier_feature_map",
    "hea",
    "identity_initialized_ansatz",
    "build_qnn",
    "hamiltonian_factory",
    "ising_hamiltonian",
    "single_z",
    "total_magnetization",
    "zz_hamiltonian",
    "qft",
    "daqc_transform",
    "rydberg_hea",
    "rydberg_hea_layer",
    "rydberg_feature_map",
    "analog_feature_map",
    "rydberg_tower_feature_map",
]
