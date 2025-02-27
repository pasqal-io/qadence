from __future__ import annotations

from dataclasses import dataclass, field, fields
from logging import getLogger
from pathlib import Path
from typing import Callable, Type

from sympy import Basic

from qadence.blocks.analog import AnalogBlock
from qadence.blocks.primitive import ParametricBlock
from qadence.operations import RX, AnalogRX
from qadence.parameters import Parameter
from qadence.types import (
    AnsatzType,
    BasisSet,
    ExperimentTrackingTool,
    LoggablePlotFunction,
    MultivariateStrategy,
    ReuploadScaling,
    Strategy,
)
from torch import dtype

logger = getLogger(__file__)


@dataclass
class TrainConfig:
    """Default configuration for the training process.

    This class provides default settings for various aspects of the training loop,
    such as logging, checkpointing, and validation. The default values for these
    fields can be customized when an instance of `TrainConfig` is created.

    Example:
    ```python exec="on" source="material-block" result="json"
    from qadence.ml_tools import TrainConfig
    c = TrainConfig(root_folder="/tmp/train")
    print(str(c)) # markdown-exec: hide
    ```
    """

    max_iter: int = 10000
    """Number of training iterations (epochs) to perform.

    This defines the total number
    of times the model will be updated.

    In case of InfiniteTensorDataset, each epoch will have 1 batch.
    In case of TensorDataset, each epoch will have len(dataloader) batches.
    """

    print_every: int = 0
    """Frequency (in epochs) for printing loss and metrics to the console during training.

    Set to 0 to disable this output, meaning that metrics and loss will not be printed
    during training.
    """

    write_every: int = 0
    """Frequency (in epochs) for writing loss and metrics using the tracking tool during training.

    Set to 0 to disable this logging, which prevents metrics from being logged to the tracking tool.
    Note that the metrics will always be written at the end of training regardless of this setting.
    """

    checkpoint_every: int = 0
    """Frequency (in epochs) for saving model and optimizer checkpoints during training.

    Set to 0 to disable checkpointing. This helps in resuming training or recovering
    models.
    Note that setting checkpoint_best_only = True will disable this and only best checkpoints will
    be saved.
    """

    plot_every: int = 0
    """Frequency (in epochs) for generating and saving figures during training.

    Set to 0 to disable plotting.
    """

    callbacks: list = field(default_factory=lambda: list())
    """List of callbacks to execute during training.

    Callbacks can be used for
    custom behaviors, such as early stopping, custom logging, or other actions
    triggered at specific events.
    """

    log_model: bool = False
    """Whether to log a serialized version of the model.

    When set to `True`, the
    model's state will be logged, useful for model versioning and reproducibility.
    """

    root_folder: Path = Path("./qml_logs")
    """The root folder for saving checkpoints and tensorboard logs.

    The default path is "./qml_logs"

    This can be set to a specific directory where training artifacts are to be stored.
    Checkpoints will be saved inside a subfolder in this directory. Subfolders will be
    created based on `create_subfolder_per_run` argument.
    """

    create_subfolder_per_run: bool = False
    """Whether to create a subfolder for each run, named `<id>_<timestamp>_<PID>`.

    This ensures logs and checkpoints from different runs do not overwrite each other,
    which is helpful for rapid prototyping. If `False`, training will resume from
    the latest checkpoint if one exists in the specified log folder.
    """

    log_folder: Path = Path("./")
    """The log folder for saving checkpoints and tensorboard logs.

    This stores the path where all logs and checkpoints are being saved
    for this training session. `log_folder` takes precedence over `root_folder`,
    but it is ignored if `create_subfolders_per_run=True` (in which case, subfolders
    will be spawned in the root folder).
    """

    checkpoint_best_only: bool = False
    """If `True`, checkpoints are only saved if there is an improvement in the.

    validation metric. This conserves storage by only keeping the best models.

    validation_criterion is required when this is set to True.
    """

    val_every: int = 0
    """Frequency (in epochs) for performing validation.

    If set to 0, validation is not performed.
    Note that metrics from validation are always written, regardless of the `write_every` setting.
    Note that initial validation happens at the start of training (when val_every > 0)
        For initial validation  - initial metrics are written.
                                - checkpoint is saved (when checkpoint_best_only = False)
    """

    val_epsilon: float = 1e-5
    """A small safety margin used to compare the current validation loss with the.

    best previous validation loss. This is used to determine improvements in metrics.
    """

    validation_criterion: Callable | None = None
    """A function to evaluate whether a given validation metric meets a desired condition.

    The validation_criterion has the following format:
    def validation_criterion(val_loss: float, best_val_loss: float, val_epsilon: float) -> bool:
        # process

    If `None`, no custom validation criterion is applied.
    """

    trainstop_criterion: Callable | None = None
    """A function to determine if the training process should stop based on a.

    specific stopping metric. If `None`, training continues until `max_iter` is reached.
    """

    batch_size: int = 1
    """The batch size to use when processing a list or tuple of torch.Tensors.

    This specifies how many samples are processed in each training iteration.
    """

    verbose: bool = True
    """Whether to print metrics and status messages during training.

    If `True`, detailed metrics and status updates will be displayed in the console.
    """

    tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
    """The tool used for tracking training progress and logging metrics.

    Options include tools like TensorBoard, which help visualize and monitor
    model training.
    """

    hyperparams: dict = field(default_factory=dict)
    """A dictionary of hyperparameters to be tracked.

    This can include learning rates,
    regularization parameters, or any other training-related configurations.
    """

    plotting_functions: tuple[LoggablePlotFunction, ...] = field(default_factory=tuple)  # type: ignore
    """Functions used for in-training plotting.

    These are called to generate
    plots that are logged or saved at specified intervals.
    """

    _subfolders: list[str] = field(default_factory=list)
    """List of subfolders used for logging different runs using the same config inside the.

    root folder.

    Each subfolder is of structure `<id>_<timestamp>_<PID>`.
    """

    nprocs: int = 1
    """
    The number of processes to use for training when spawning subprocesses.

    For effective parallel processing, set this to a value greater than 1.
    - In case of Multi-GPU or Multi-Node-Multi-GPU setups, nprocs should be equal to
    the total number of GPUs across all nodes (world size), or total number of GPU to be used.

    If nprocs > 1, multiple processes will be spawned for training. The training framework will launch
    additional processes (e.g., for distributed or parallel training).
    - For CPU setup, this will launch a true parallel processes
    - For GPU setup, this will launch a distributed training routine.
    This uses the DistributedDataParallel framework from PyTorch.
    """

    compute_setup: str = "cpu"
    """
    Compute device setup; options are "auto", "gpu", or "cpu".

    - "auto": Automatically uses GPU if available; otherwise, falls back to CPU.
    - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
    - "cpu": Forces the use of CPU regardless of GPU availability.
    """

    backend: str = "gloo"
    """
    Backend used for distributed training communication.

    The default is "gloo". Other options may include "nccl" - which is optimized for GPU-based training or "mpi",
    depending on your system and requirements.
    It should be one of the backends supported by `torch.distributed`. For further details, please look at
    [torch backends](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend)
    """

    log_setup: str = "cpu"
    """
    Logging device setup; options are "auto" or "cpu".

    - "auto": Uses the same device for logging as for computation.
    - "cpu": Forces logging to occur on the CPU. This can be useful to avoid potential conflicts with GPU processes.
    """

    dtype: dtype | None = None
    """
    Data type (precision) for computations.

    Both model parameters, and dataset will be of the provided precision.

    If not specified or None, the default torch precision (usually torch.float32) is used.
    If provided dtype is torch.complex128, model parameters will be torch.complex128, and data parameters will be torch.float64
    """

    all_reduce_metrics: bool = False
    """
    Whether to aggregate metrics (e.g., loss, accuracy) across processes.

    When True, metrics from different training processes are averaged to provide a consolidated metrics.
    Note: Since aggregation requires synchronization/all_reduce operation, this can increase the
     computation time significantly.
    """


@dataclass
class FeatureMapConfig:
    num_features: int = 0
    """
    Number of feature parameters to be encoded.

    Defaults to 0. Thus, no feature parameters are encoded.
    """

    basis_set: BasisSet | dict[str, BasisSet] = BasisSet.FOURIER
    """
    Basis set for feature encoding.

    Takes qadence.BasisSet.
    Give a single BasisSet to use the same for all features.
    Give a dict of (str, BasisSet) where the key is the name of the variable and the
    value is the BasisSet to use for encoding that feature.
    BasisSet.FOURIER for Fourier encoding.
    BasisSet.CHEBYSHEV for Chebyshev encoding.
    """

    reupload_scaling: ReuploadScaling | dict[str, ReuploadScaling] = ReuploadScaling.CONSTANT
    """
    Scaling for encoding the same feature on different qubits.

    Scaling used to encode the same feature on different qubits in the
    same layer of the feature maps. Takes qadence.ReuploadScaling.
    Give a single ReuploadScaling to use the same for all features.
    Give a dict of (str, ReuploadScaling) where the key is the name of the variable and the
    value is the ReuploadScaling to use for encoding that feature.
    ReuploadScaling.CONSTANT for constant scaling.
    ReuploadScaling.TOWER for linearly increasing scaling.
    ReuploadScaling.EXP for exponentially increasing scaling.
    """

    feature_range: tuple[float, float] | dict[str, tuple[float, float]] | None = None
    """
    Range of data that the input data is assumed to come from.

    Give a single tuple to use the same range for all features.
    Give a dict of (str, tuple) where the key is the name of the variable and the
    value is the feature range to use for that feature.
    """

    target_range: tuple[float, float] | dict[str, tuple[float, float]] | None = None
    """
    Range of data the data encoder assumes as natural range.

    Give a single tuple to use the same range for all features.
    Give a dict of (str, tuple) where the key is the name of the variable and the
    value is the target range to use for that feature.
    """

    multivariate_strategy: MultivariateStrategy = MultivariateStrategy.PARALLEL
    """
    The encoding strategy in case of multi-variate function.

    Takes qadence.MultivariateStrategy.
    If PARALLEL, the features are encoded in one block of rotation gates
    with the register being split in sub-registers for each feature.
    If SERIES, the features are encoded sequentially using the full register for each feature, with
    an ansatz block between them. PARALLEL is allowed only for DIGITAL `feature_map_strategy`.
    """

    feature_map_strategy: Strategy = Strategy.DIGITAL
    """
    Strategy for feature map.

    Accepts DIGITAL, ANALOG or RYDBERG. Defaults to DIGITAL.
    If the strategy is incompatible with the `operation` chosen, then `operation`
    gets preference and the given strategy is ignored.
    """

    param_prefix: str | None = None
    """
    String prefix to create trainable parameters in Feature Map.

    A string prefix to create trainable parameters multiplying the feature parameter
    inside the feature-encoding function. Note that currently this does not take into
    account the domain of the feature-encoding function.
    Defaults to `None` and thus, the feature map is not trainable.
    Note that this is separate from the name of the parameter.
    The user can provide a single prefix for all features, and it will be appended
    by appropriate feature name automatically.
    """

    num_repeats: int | dict[str, int] = 0
    """
    Number of feature map layers repeated in the data reuploading step.

    If all features are to be repeated the same number of times, then can give a single
    `int`. For different number of repetitions for each feature, provide a dict
    of (str, int) where the key is the name of the variable and the value is the
    number of repetitions for that feature.
    This amounts to the number of additional reuploads. So if `num_repeats` is N,
    the data gets uploaded N+1 times. Defaults to no repetition.
    """

    operation: Callable[[Parameter | Basic], AnalogBlock] | Type[RX] | None = None
    """
    Type of operation.

    Choose among the analog or digital rotations or a custom
    callable function returning an AnalogBlock instance. If the type of operation is
    incompatible with the `strategy` chosen, then `operation` gets preference and
    the given strategy is ignored.
    """

    inputs: list[Basic | str] | None = None
    """
    List that indicates the order of variables of the tensors that are passed.

    Optional if a single feature is being encoded, required otherwise. Given input tensors
    `xs = torch.rand(batch_size, input_size:=2)` a QNN with `inputs=["t", "x"]` will
    assign `t, x = xs[:,0], xs[:,1]`.
    """

    tag: str | None = None
    """
    String to indicate the name tag of the feature map.

    Defaults to None, in which case no tag will be applied.
    """

    def __post_init__(self) -> None:
        if self.multivariate_strategy == MultivariateStrategy.PARALLEL and self.num_features > 1:
            assert (
                self.feature_map_strategy == Strategy.DIGITAL
            ), "For parallel encoding of multiple features, the `feature_map_strategy` must be \
                  of `Strategy.DIGITAL`."

        if self.operation is None:
            if self.feature_map_strategy == Strategy.DIGITAL:
                self.operation = RX
            elif self.feature_map_strategy == Strategy.ANALOG:
                self.operation = AnalogRX  # type: ignore[assignment]

        else:
            if self.feature_map_strategy == Strategy.DIGITAL:
                if isinstance(self.operation, AnalogBlock):
                    logger.warning(
                        "The `operation` is of type `AnalogBlock` but the `feature_map_strategy` is\
                        `Strategy.DIGITAL`. The `feature_map_strategy` will be modified and given \
                        operation will be used."
                    )

                    self.feature_map_strategy = Strategy.ANALOG

            elif self.feature_map_strategy == Strategy.ANALOG:
                if isinstance(self.operation, ParametricBlock):
                    logger.warning(
                        "The `operation` is a digital gate but the `feature_map_strategy` is\
                        `Strategy.ANALOG`. The `feature_map_strategy` will be modified and given\
                        operation will be used."
                    )

                    self.feature_map_strategy = Strategy.DIGITAL

            elif self.feature_map_strategy == Strategy.RYDBERG:
                if self.operation is not None:
                    logger.warning(
                        f"feature_map_strategy is `Strategy.RYDBERG` which does not take any\
                        operation. But an operation {self.operation} is provided. The \
                        `feature_map_strategy` will be modified and given operation will be used."
                    )

                    if isinstance(self.operation, AnalogBlock):
                        self.feature_map_strategy = Strategy.ANALOG
                    else:
                        self.feature_map_strategy = Strategy.DIGITAL

        if self.inputs is not None:
            assert (
                len(self.inputs) == self.num_features
            ), "Inputs list must be of same size as the number of features"
        else:
            if self.num_features == 0:
                self.inputs = []
            elif self.num_features == 1:
                self.inputs = ["x"]
            else:
                raise ValueError(
                    """
                    Your QNN has more than one input. Please provide a list of inputs in the order
                    of your tensor domain. For example, if you want to pass
                    `xs = torch.rand(batch_size, input_size:=3)` to you QNN, where
                    ```
                    t = x[:,0]
                    x = x[:,1]
                    y = x[:,2]
                    ```
                    you have to specify
                    ```
                    inputs=["t", "x", "y"]
                    ```
                    You can also pass a list of sympy symbols.
                """
                )

        property_list = [
            "basis_set",
            "reupload_scaling",
            "feature_range",
            "target_range",
            "num_repeats",
        ]

        for target_field in fields(self):
            if target_field.name in property_list:
                prop = getattr(self, target_field.name)
                if isinstance(prop, dict):
                    assert set(prop.keys()) == set(
                        self.inputs
                    ), f"The keys in {target_field.name} must be the same as the inputs provided. \
                    Alternatively, provide a single value of {target_field.name} to use the same\
                    {target_field.name} for all features."
                else:
                    prop = {key: prop for key in self.inputs}
                    setattr(self, target_field.name, prop)


@dataclass
class AnsatzConfig:
    depth: int = 1
    """Number of layers of the ansatz."""

    ansatz_type: AnsatzType = AnsatzType.HEA
    """What type of ansatz.

    `AnsatzType.HEA` for Hardware Efficient Ansatz.
    `AnsatzType.IIA` for Identity Intialized Ansatz.
    `AnsatzType.ALA` for Alternating Layer Ansatz.
    """

    ansatz_strategy: Strategy = Strategy.DIGITAL
    """Ansatz strategy.

    `Strategy.DIGITAL` for fully digital ansatz. Required if `ansatz_type` is `AnsatzType.ALA`.
    `Strategy.SDAQC` for analog entangling block. Only available for `AnsatzType.HEA` or
    `AnsatzType.ALA`.
    `Strategy.RYDBERG` for fully rydberg hea ansatz. Only available for `AnsatzType.HEA`.
    """

    strategy_args: dict = field(default_factory=dict)
    """
    A dictionary containing keyword arguments to the function creating the ansatz.

    Details about each below.

    For `Strategy.DIGITAL` strategy, accepts the following:
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
            Defaults to  [RX, RY, RX] for hea and [RX, RY] for iia.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ, CPHASE. Controlld rotations
            will have variational parameters on the rotation angles.
            Defaults to CNOT

    For `Strategy.SDAQC` strategy, accepts the following:
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
            Defaults to  [RX, RY, RX] for hea and [RX, RY] for iia.
        entangler (AbstractBlock): Hamiltonian generator for the
            analog entangling layer. Time parameter is considered variational.
            Defaults to NN interaction.

    For `Strategy.RYDBERG` strategy, accepts the following:
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

    m_block_qubits: int | None = None
    """
    The number of qubits in the local entangling block of an Alternating Layer Ansatz (ALA).

    Only used when `ansatz_type` is `AnsatzType.ALA`.
    """

    param_prefix: str = "theta"
    """The base bame of the variational parameter."""

    tag: str | None = None
    """
    String to indicate the name tag of the ansatz.

    Defaults to None, in which case no tag will be applied.
    """

    def __post_init__(self) -> None:
        if self.ansatz_type == AnsatzType.IIA:
            assert (
                self.ansatz_strategy != Strategy.RYDBERG
            ), "Rydberg strategy not allowed for Identity-initialized ansatz."

        if self.ansatz_type == AnsatzType.ALA:
            assert (
                self.ansatz_strategy == Strategy.DIGITAL
            ), f"{self.ansatz_strategy} not allowed for Alternating Layer Ansatz.\
            Only `Strategy.DIGITAL` allowed."

            assert (
                self.m_block_qubits is not None
            ), "m_block_qubits must be specified for Alternating Layer Ansatz."
