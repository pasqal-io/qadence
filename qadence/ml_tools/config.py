from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Type

from sympy import Basic

from qadence.blocks.analog import AnalogBlock
from qadence.blocks.primitive import ParametricBlock
from qadence.logger import get_logger
from qadence.operations import RX, AnalogRX
from qadence.parameters import Parameter
from qadence.types import BasisSet, ReuploadScaling

logger = get_logger(__file__)


@dataclass
class TrainConfig:
    """Default config for the train function.

    The default value of
    each field can be customized with the constructor:

    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    c = TrainConfig(folder="/tmp/train")
    print(str(c)) # markdown-exec: hide
    ```
    """

    max_iter: int = 10000
    """Number of training iterations."""
    print_every: int = 1000
    """Print loss/metrics."""
    write_every: int = 50
    """Write tensorboard logs."""
    checkpoint_every: int = 5000
    """Write model/optimizer checkpoint."""
    folder: Optional[Path] = None
    """Checkpoint/tensorboard logs folder."""
    create_subfolder_per_run: bool = False
    """Checkpoint/tensorboard logs stored in subfolder with name `<timestamp>_<PID>`.

    Prevents continuing from previous checkpoint, useful for fast prototyping.
    """
    checkpoint_best_only: bool = False
    """Write model/optimizer checkpoint only if a metric has improved."""
    validation_criterion: Optional[Callable] = None
    """A boolean function which evaluates a given validation metric is satisfied."""
    trainstop_criterion: Optional[Callable] = None
    """A boolean function which evaluates a given training stopping metric is satisfied."""
    batch_size: int = 1
    """The batch_size to use when passing a list/tuple of torch.Tensors."""
    verbose: bool = True
    """Whether or not to print out metrics values during training."""

    def __post_init__(self) -> None:
        if self.folder:
            if isinstance(self.folder, str):  # type: ignore [unreachable]
                self.folder = Path(self.folder)  # type: ignore [unreachable]
            if self.create_subfolder_per_run:
                subfoldername = (
                    datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + hex(os.getpid())[2:]
                )
                self.folder = self.folder / subfoldername
        if self.trainstop_criterion is None:
            self.trainstop_criterion = lambda x: x <= self.max_iter
        if self.validation_criterion is None:
            self.validation_criterion = lambda x: False


@dataclass
class FeatureMapConfig:
    num_features: int
    """Number of feature parameters to be encoded."""

    basis_set: BasisSet | list[BasisSet]
    """
    Basis set for feature encoding.

    Takes qadence.BasisSet.
    Give a single BasisSet to use the same for all features.
    Give a list of BasisSet to apply each item for a corresponding variable.
    BasisSet.FOURIER for Fourier encoding.
    BasisSet.CHEBYSHEV for Chebyshev encoding.
    """

    reupload_scaling: ReuploadScaling | list[ReuploadScaling]
    """
    Scaling for encoding the same feature on different qubits in the.

    same layer of the feature maps. Takes qadence.ReuploadScaling.
    Give a single ReuploadScaling to use the same for all features.
    Give a list of ReuploadScaling to apply each item for a corresponding variable.
    ReuploadScaling.CONSTANT for constant scaling.
    ReuploadScaling.TOWER for linearly increasing scaling.
    ReuploadScaling.EXP for exponentially increasing scaling.
    """

    feature_range: tuple[float, float] | list[tuple[float, float]]
    """
    Range of data that the input data is assumed to come from.

    Give a single tuple to use the same range for all features.
    Give a list of tuples to use each item for a corresponding variable.
    """

    target_range: tuple[float, float] | list[tuple[float, float]] | None = None
    """
    Range of data the data encoder assumes as natural range.

    Give a single tuple to use the same range for all features.
    Give a list of tuples to use each item for a corresponding variable.
    """

    multivariate_strategy: str = "parallel"
    """
    The  encoding strategy in case of multi-variate function.

    If "parallel",
    the features are encoded in one block of rotation gates. with each
    feature given an equal number of qubits. If "serial", the features are
    encoded sequentially, with a HEA block between. "parallel" is allowed
    only for "digital" `feature_map_strategy`.
    """

    feature_map_strategy: str = "digital"
    """Strategy for feature map.

    Accepts 'digital', 'analog' or 'rydberg'. Defaults to "digital".
    """

    param_prefix: str = "phi"
    """The base name of the feature map parameter.

    Defaults to `phi`.
    """

    num_repeats: int | list[int] = 0
    """
    Number of feature map layers repeated in the data reuploadig step.

    If all are to be repeated the same number of times, then can give a single
    `int`. For different number of repeatitions for each feature, provide a list
    of `int`s corresponding to desired number of repeatitions. This amounts to
    the number of additional reuploads. So if `num_repeats` is N, the data gets
    uploaded N+1 times. Defaults to no repeatition.
    """

    operation: Callable[[Parameter | Basic], AnalogBlock] | Type[RX] | None = None
    """
    Type of operation.

    Choose among the analog or digital rotations or a custom
    callable function returning an AnalogBlock instance
    """

    inputs: list[Basic | str] | None = None
    """
    List that indicates the order of variables of the tensors that are passed.

    to the model. Given input tensors `xs = torch.rand(batch_size, input_size:=2)` a QNN
    with `inputs=("t", "x")` will assign `t, x = xs[:,0], xs[:,1]`.
    """

    def __post_init__(self) -> None:
        if self.multivariate_strategy == "parallel" and self.num_features > 1:
            assert (
                self.feature_map_strategy == "digital"
            ), "For `parallel` encoding of multiple features, the `feature_map_strategy` must be \
                  of `digital` type."

        if self.operation is None:
            if self.feature_map_strategy == "digital":
                self.operation = RX
            elif self.feature_map_strategy == "analog":
                self.operation = AnalogRX  # type: ignore[assignment]

        else:
            if self.feature_map_strategy == "digital":
                if isinstance(self.operation, AnalogBlock):
                    logger.warning(
                        "The `operation` is of type `AnalogBlock` but the `feature_map_strategy` is\
                        `digital`. The `feature_map_strategy` will be modified and given operation\
                        will be used."
                    )

                    self.feature_map_strategy = "analog"

            elif self.feature_map_strategy == "analog":
                if isinstance(self.operation, ParametricBlock):
                    logger.warning(
                        "The `operation` is a digital gate but the `feature_map_strategy` is\
                        `analog`. The `feature_map_strategy` will be modified and given operation\
                        will be used."
                    )

                    self.feature_map_strategy = "digital"

        if isinstance(self.basis_set, BasisSet):
            self.basis_set = [self.basis_set for i in range(self.num_features)]
        else:
            assert (
                len(self.basis_set) == self.num_features
            ), f"Length of set of bases {len(self.feature_range)} must match the number \
                of features {self.num_features}. Or provide a single `BasisSet` to \
                use same basis for all features."

        if isinstance(self.reupload_scaling, ReuploadScaling):
            self.reupload_scaling = [self.reupload_scaling for i in range(self.num_features)]
        else:
            assert (
                len(self.reupload_scaling) == self.num_features
            ), f"Length of the reupload scalings {len(self.feature_range)} must match the number \
                of features {self.num_features}. Or provide a single `ReuploadScaling` for \
                same scaling for all features."

        if isinstance(self.feature_range, tuple):
            self.feature_range = [self.feature_range for i in range(self.num_features)]
        else:
            assert (
                len(self.feature_range) == self.num_features
            ), f"Length of the feature ranges {len(self.feature_range)} must match the number \
                of features {self.num_features}. Or provide a single tuple(float, float) for \
                same expected feature range for all features."

        if self.target_range is None:
            self.target_range = [None for i in range(self.num_features)]  # type: ignore
        elif isinstance(self.target_range, tuple):
            self.target_range = [self.target_range for i in range(self.num_features)]
        else:
            assert (
                len(self.target_range) == self.num_features
            ), f"Length of the feature ranges {len(self.feature_range)} must match the number \
                of features {self.num_features}. Or provide a single tuple(float, float) for \
                same expected feature range for all features."

        if isinstance(self.num_repeats, int):
            self.num_repeats = [self.num_repeats for i in range(self.num_features)]
        else:
            assert (
                len(self.num_repeats) == self.num_features
            ), f"Length of the repeat array {len(self.num_repeats)} must match the number \
                of features {self.num_features}. Or provide a single integer for same number \
                of repeatitions for all features."

        if self.inputs is None:
            self.inputs = [f"phi_{i}" for i in range(self.num_features)]


@dataclass
class AnsatzConfig:
    num_layers: int
    """Number of layers of the ansatz."""

    ansatz_type: str
    """What type of ansatz.

    "hea" for Hardware Efficient Ansatz.
    "iia" for Identity intialized Ansatz.
    """

    ansatz_strategy: str
    """Ansatz strategy.

    "digital" for fully digital ansatz. Required if `ansatz_type` is `iia`.
    "sdaqc" for analog entangling block.
    "rydberg" for fully rydberg hea ansatz.
    """

    strategy_args: dict = field(default_factory=dict)
    """
    A dictionary containing keyword arguments to the function creating the ansatz.

    Details about each below.

    For "digital" strategy, accepts the following:
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
            Defaults to  [RX, RY, RX] for hea and [RX, RY] for iia.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ, CPHASE. Controlld rotations
            will have variational parameters on the rotation angles.
            Defaults to CNOT

    For "sdaqc" strategy, accepts the following:
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
            Defaults to  [RX, RY, RX] for hea and [RX, RY] for iia.
        entangler (AbstractBlock): Hamiltonian generator for the
            analog entangling layer. Time parameter is considered variational.
            Defaults to NN interaction.

    For "rydberg" strategy, accepts the following:
        addressable_detuning: whether to turn on the trainable semi-local addressing pattern
            on the detuning (n_i terms in the Hamiltonian).
            Defaults to True.
        addressable_drive: whether to turn on the trainable semi-local addressing pattern
            on the drive (sigma_i^x terms in the Hamiltonian).
            Defaults to False.
        tunable_phase: whether to have a tunable phase to get both sigma^x and sigma^y rotations
            in the drive term. If False, only a sigma^x term will be included in the drive part
            of the Hamiltonian generator.
            Defaults to False.
    """
    # The default for a dataclass can not be a mutable object without using this default_factory.

    param_prefix: str = "theta"
    """The base bame of the variational parameter."""

    def __post_init__(self) -> None:
        if self.ansatz_type == "iia":
            assert (
                self.ansatz_strategy != "rydberg"
            ), "Rydberg strategy not allowed for Identity-initialized ansatz."
