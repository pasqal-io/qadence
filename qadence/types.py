from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import sympy
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from pyqtorch.noise import NoiseType as DigitalNoise
from pyqtorch.utils import SolverType
from torch import Tensor, pi
from torch.nn import Module

TNumber = Union[int, float, complex, np.int64, np.float64]
"""Union of python and numpy numeric types."""

TDrawColor = Tuple[float, float, float, float]

TParameter = Union[TNumber, Tensor, sympy.Basic, str]
"""Union of numbers, tensors, and parameter types."""

TArray = Union[Iterable, Tensor, np.ndarray]
"""Union of common array types."""

TGenerator = Union[Tensor, sympy.Array, sympy.Basic]
"""Union of torch tensors and numpy arrays."""


PI = pi

# Modules to be automatically added to the qadence namespace
__all__ = [
    "AnsatzType",
    "Endianness",
    "Strategy",
    "ResultType",
    "ParameterType",
    "BackendName",
    "StateGeneratorType",
    "LTSOrder",
    "MultivariateStrategy",
    "ReuploadScaling",
    "BasisSet",
    "TensorType",
    "DiffMode",
    "BackendName",
    "Interaction",
    "DeviceType",
    "OverlapMethod",
    "AlgoHEvo",
    "SerializationFormat",
    "PI",
    "SolverType",
    "NoiseProtocol",
]  # type: ignore


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class Strategy(StrEnum):
    """Computing paradigm."""

    DIGITAL = "Digital"
    """Use the digital paradigm."""
    ANALOG = "Analog"
    """Use the analog paradigm."""
    SDAQC = "sDAQC"
    """Use the step-wise digital-analog QC paradigm."""
    BDAQC = "bDAQC"
    """Use the banged digital-analog QC paradigm."""
    RYDBERG = "Rydberg"
    """Use the Rydberg QC paradigm."""


class Endianness(StrEnum):
    """The endianness convention to use."""

    BIG = "Big"
    """Use Big endianness."""
    LITTLE = "Little"
    """Use little endianness."""


class ResultType(StrEnum):
    """Available data types for generating certain results."""

    STRING = "String"
    """String Type."""
    TORCH = "Torch"
    """Torch Tensor Type."""
    NUMPY = "Numpy"
    """Numpy Array Type."""


class ParameterType(StrEnum):
    """Parameter types available in qadence."""

    FEATURE = "Feature"
    """FeatureParameters act as input and are not trainable."""
    VARIATIONAL = "Variational"
    """VariationalParameters are trainable."""
    FIXED = "Fixed"
    """Fixed/ constant parameters are neither trainable nor act as input."""


class TensorType(StrEnum):
    """Tensor Types for converting blocks to tensors."""

    SPARSEDIAGONAL = "SparseDiagonal"
    """Convert a diagonal observable block to a sparse diagonal if possible."""
    DENSE = "Dense"
    """Convert a block to a dense tensor."""
    SPARSE = "Sparse"
    """Convert a observable block to a sparse tensor."""


class LTSOrder(StrEnum):
    """Lie-Trotter-Suzuki approximation order."""

    BASIC = "BASIC"
    """Basic."""
    ST2 = "ST2"
    """ST2."""
    ST4 = "ST4"
    """ST4."""


class BasisSet(StrEnum):
    """Basis set for feature maps."""

    FOURIER = "Fourier"
    """Fourier basis set."""
    CHEBYSHEV = "Chebyshev"
    """Chebyshev polynomials of the first kind."""


class ReuploadScaling(StrEnum):
    """Scaling for data reuploads in feature maps."""

    CONSTANT = "Constant"
    """Constant scaling."""
    TOWER = "Tower"
    """Linearly increasing scaling."""
    EXP = "Exponential"
    """Exponentially increasing scaling."""


class MultivariateStrategy(StrEnum):
    """Multivariate strategy for feature maps."""

    PARALLEL = "Parallel"
    """Parallel strategy."""
    SERIES = "Series"
    """Serial strategy."""


class AnsatzType(StrEnum):
    """Ansatz types for variational circuits."""

    HEA = "hea"
    """Hardware-efficient ansatz."""
    IIA = "iia"
    """Identity-Initialised Ansatz."""


class _DiffMode(StrEnum):
    """Differentiation modes to choose from."""

    GPSR = "gpsr"
    """Basic generalized parameter shift rule."""
    AD = "ad"
    """Automatic Differentiation."""
    ADJOINT = "adjoint"
    """Adjoint Differentiation."""


class QubitSupportType(StrEnum):
    """Qubit support types."""

    GLOBAL = "global"
    """Use global qubit support."""


class Interaction(StrEnum):
    """Interaction types used in.

    - `RydbergDevice`.
    - [`hamiltonian_factory`][qadence.constructors.hamiltonians.hamiltonian_factory].
    """

    ZZ = "ZZ"
    """ZZ-Ising Interaction."""
    NN = "NN"
    """NN-Ising Interaction, N=(I-Z)/2."""
    XY = "XY"
    """XY Interaction."""
    XYZ = "XYZ"
    """XYZ Interaction."""


class DeviceType(StrEnum):
    """Supported types of devices for Pulser backend."""

    IDEALIZED = "IdealDevice"
    """Idealized device, least realistic."""

    REALISTIC = "RealisticDevice"
    """Device with realistic specs."""


class _BackendName(StrEnum):
    """The available backends for running circuits."""

    PYQTORCH = "pyqtorch"
    """The Pyqtorch backend."""
    PULSER = "pulser"
    """The Pulser backend."""
    HORQRUX = "horqrux"
    """The horqrux backend."""


class _Engine(StrEnum):
    TORCH = "torch"
    JAX = "jax"


# If proprietary qadence_extensions is available, import the
# right function since more backends are supported.
try:
    module = importlib.import_module("qadence_extensions.types")
    BackendName = getattr(module, "BackendName")
    DiffMode = getattr(module, "DiffMode")
    Engine = getattr(module, "Engine")
except ModuleNotFoundError:
    BackendName = _BackendName
    DiffMode = _DiffMode
    Engine = _Engine


class StateGeneratorType(StrEnum):
    """Methods to generate random states."""

    RANDOM_ROTATIONS = "RandomRotations"
    """Random Rotations."""
    HAAR_MEASURE_FAST = "HaarMeasureFast"
    """HaarMeasure."""
    HAAR_MEASURE_SLOW = "HaarMeasureSlow"
    """HaarMeasure non-optimized version."""


class SerializationFormat(StrEnum):
    """Available serialization formats for circuits."""

    PT = "PT"
    """The PT format used by Torch."""
    JSON = "JSON"
    """The Json format."""


class OverlapMethod(StrEnum):
    """Overlap Methods to choose from."""

    EXACT = "exact"
    """Exact."""
    JENSEN_SHANNON = "jensen_shannon"
    """Jensen-shannon."""
    COMPUTE_UNCOMPUTE = "compute_uncompute"
    """Compute-uncompute."""
    SWAP_TEST = "swap_test"
    """Swap-test."""
    HADAMARD_TEST = "hadamard_test"
    """Hadamard-test."""


class FigFormat(StrEnum):
    """Available output formats for exporting visualized circuits to a file."""

    PNG = "PNG"
    """PNG format."""
    PDF = "PDF"
    """PDF format."""
    SVG = "SVG"
    """SVG format."""


class AlgoHEvo(StrEnum):
    """Hamiltonian Evolution algorithms that can be used by the backend."""

    RK4 = "RK4"
    """4th order Runge-Kutta approximation."""
    EIG = "EIG"
    """Using Hamiltonian diagonalization."""
    EXP = "EXP"
    """Using torch.matrix_exp on the generator matrix."""


class LatticeTopology(StrEnum):
    """Lattice topologies to choose from for the register."""

    LINE = "line"
    """Line-format lattice."""
    SQUARE = "square"
    """Square lattice."""
    CIRCLE = "circle"
    """Circular lattice."""
    ALL_TO_ALL = "all_to_all"
    """All to all- connected lattice."""
    RECTANGULAR_LATTICE = "rectangular_lattice"
    """Rectangular-shaped lattice."""
    TRIANGULAR_LATTICE = "triangular_lattice"
    """Triangular-shaped shape."""
    HONEYCOMB_LATTICE = "honeycomb_lattice"
    """Honeycomb-shaped lattice."""
    ARBITRARY = "arbitrary"
    """Arbitrarily-shaped lattice."""


class GenDAQC(StrEnum):
    """The type of interaction for the DAQC transform."""

    ZZ = "ZZ"
    """ZZ"""
    NN = "NN"
    """NN"""


class OpName(StrEnum):
    """A list of all available of digital-analog operations."""

    # Digital operations
    X = "X"
    """The X gate."""
    Y = "Y"
    """The Y gate."""
    Z = "Z"
    """The Z gate."""
    N = "N"
    """The N = (1/2)(I-Z) operator."""
    H = "H"
    """The Hadamard gate."""
    I = "I"  # noqa
    """The Identity gate."""
    ZERO = "Zero"
    """The zero gate."""
    RX = "RX"
    """The RX gate."""
    RY = "RY"
    """The RY gate."""
    RZ = "RZ"
    """The RZ gate."""
    U = "U"
    """The U gate."""
    CNOT = "CNOT"
    """The CNOT gate."""
    CZ = "CZ"
    """The CZ gate."""
    MCZ = "MCZ"
    """The Multicontrol CZ gate."""
    HAMEVO = "HamEvo"
    """The Hamiltonian Evolution operation."""
    CRX = "CRX"
    """The Control RX gate."""
    MCRX = "MCRX"
    """The Multicontrol RX gate."""
    CRY = "CRY"
    """The Controlled RY gate."""
    MCRY = "MCRY"
    """The Multicontrol RY gate."""
    CRZ = "CRZ"
    """The Control RZ gate."""
    MCRZ = "MCRZ"
    """The Multicontrol RZ gate."""
    CSWAP = "CSWAP"
    """The Control SWAP gate."""
    T = "T"
    """The T gate."""
    # FIXME: Tdagger is not currently supported by any backend
    TDAGGER = "TDagger"
    """The T dagger gate."""
    S = "S"
    """The S gate."""
    SDAGGER = "SDagger"
    """The S dagger gate."""
    SWAP = "SWAP"
    """The SWAP gate."""
    PHASE = "PHASE"
    """The PHASE gate."""
    CPHASE = "CPHASE"
    """The controlled PHASE gate."""
    MCPHASE = "MCPHASE"
    """The Multicontrol PHASE gate."""
    TOFFOLI = "Toffoli"
    """The Toffoli gate."""
    # Analog operations
    ANALOGENTANG = "AnalogEntanglement"
    """The analog entanglement operation."""
    ANALOGRX = "AnalogRX"
    """The analog RX operation."""
    ANALOGRY = "AnalogRY"
    """The analog RY operation."""
    ANALOGRZ = "AnalogRZ"
    """The analog RZ operation."""
    ANALOGSWAP = "AnalogSWAP"
    """The analog SWAP operation."""
    ENTANGLE = "entangle"
    """The entanglement operation."""
    ANALOGINTERACTION = "AnalogInteraction"
    """The analog interaction operation."""
    PROJ = "Projector"
    """The projector operation."""


class ReadOutOptimization(StrEnum):
    MLE = "mle"
    CONSTRAINED = "constrained"


ParamDictType = dict[str, ArrayLike]
DifferentiableExpression = Callable[..., ArrayLike]


class InputDiffMode(StrEnum):
    """Derivative modes w.r.t inputs of UFAs."""

    AD = "ad"
    """Reverse automatic differentiation."""
    FD = "fd"
    """Central finite differencing."""


class ObservableTransform:
    """Observable transformation type."""

    SCALE = "scale"
    """Use the given values as scale and shift."""
    RANGE = "range"
    """Use the given values as min and max."""
    NONE = "none"
    """No transformation."""


class ExperimentTrackingTool(StrEnum):
    TENSORBOARD = "tensorboard"
    """Use the tensorboard experiment tracker."""
    MLFLOW = "mlflow"
    """Use the ml-flow experiment tracker."""


LoggablePlotFunction = Callable[[Module, int], tuple[str, Figure]]


class AnalogNoise(StrEnum):
    """Type of noise protocol."""

    DEPOLARIZING = "Depolarizing"
    DEPHASING = "Dephasing"


@dataclass
class NoiseProtocol:
    """Type of noise protocol."""

    ANALOG = AnalogNoise
    """Noise applied in analog blocks."""
    READOUT = "Readout"
    """Noise applied on outputs of quantum programs."""
    DIGITAL = DigitalNoise
    """Noise applied to digital blocks."""


NoiseEnum = Union[DigitalNoise, AnalogNoise, str]
