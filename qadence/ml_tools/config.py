from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field, fields
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Type
from uuid import uuid4

from sympy import Basic
from torch import Tensor

from qadence.blocks.analog import AnalogBlock
from qadence.blocks.primitive import ParametricBlock
from qadence.ml_tools.data import OptimizeResult
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

logger = getLogger(__file__)

CallbackFunction = Callable[[OptimizeResult], None]
CallbackConditionFunction = Callable[[OptimizeResult], bool]


class Callback:
    """Callback functions are calling in train functions.

    Each callback function should take at least as first input
    an OptimizeResult instance.

    Note: when setting call_after_opt to True, we skip
    verifying iteration % called_every == 0.

    Attributes:
        callback (CallbackFunction): Callback function accepting an
            OptimizeResult as first argument.
        callback_condition (CallbackConditionFunction | None, optional): Function that
            conditions the call to callback. Defaults to None.
        modify_optimize_result (CallbackFunction | dict[str, Any] | None, optional):
            Function that modify the OptimizeResult before callback.
            For instance, one can change the `extra` (dict) argument to be used in callback.
            If a dict is provided, the `extra` field of OptimizeResult is updated with the dict.
        called_every (int, optional): Callback to be called each `called_every` epoch.
            Defaults to 1.
            If callback_condition is None, we set
            callback_condition to returns True when iteration % called_every == 0.
        call_before_opt (bool, optional): If true, callback is applied before training.
            Defaults to False.
        call_end_epoch (bool, optional): If true, callback is applied during training,
            after an epoch is performed. Defaults to True.
        call_after_opt (bool, optional): If true, callback is applied after training.
            Defaults to False.
        call_during_eval (bool, optional): If true, callback is applied during evaluation.
            Defaults to False.
    """

    def __init__(
        self,
        callback: CallbackFunction,
        callback_condition: CallbackConditionFunction | None = None,
        modify_optimize_result: CallbackFunction | dict[str, Any] | None = None,
        called_every: int = 1,
        call_before_opt: bool = False,
        call_end_epoch: bool = True,
        call_after_opt: bool = False,
        call_during_eval: bool = False,
    ) -> None:
        """Initialized Callback.

        Args:
            callback (CallbackFunction): Callback function accepting an
                OptimizeResult as ifrst argument.
            callback_condition (CallbackConditionFunction | None, optional): Function that
                conditions the call to callback. Defaults to None.
            modify_optimize_result (CallbackFunction | dict[str, Any] | None , optional):
                Function that modify the OptimizeResult before callback. If a dict
                is provided, this updates the `extra` field of OptimizeResult.
            called_every (int, optional): Callback to be called each `called_every` epoch.
                Defaults to 1.
                If callback_condition is None, we set
                callback_condition to returns True when iteration % called_every == 0.
            call_before_opt (bool, optional): If true, callback is applied before training.
                Defaults to False.
            call_end_epoch (bool, optional): If true, callback is applied during training,
                after an epoch is performed. Defaults to True.
            call_after_opt (bool, optional): If true, callback is applied after training.
                Defaults to False.
            call_during_eval (bool, optional): If true, callback is applied during evaluation.
                Defaults to False.
        """
        self.callback = callback
        self.call_before_opt = call_before_opt
        self.call_end_epoch = call_end_epoch
        self.call_after_opt = call_after_opt
        self.call_during_eval = call_during_eval

        if called_every <= 0:
            raise ValueError("Please provide a strictly positive `called_every` argument.")
        self.called_every = called_every

        if callback_condition is None:
            self.callback_condition = lambda opt_result: True
        else:
            self.callback_condition = callback_condition

        if modify_optimize_result is None:
            self.modify_optimize_result = lambda opt_result: opt_result
        elif isinstance(modify_optimize_result, dict):

            def update_extra(opt_result: OptimizeResult) -> OptimizeResult:
                opt_result.extra.update(modify_optimize_result)
                return opt_result

            self.modify_optimize_result = update_extra
        else:
            self.modify_optimize_result = modify_optimize_result

    def __call__(self, opt_result: OptimizeResult, is_last_iteration: bool = False) -> Any:
        """Apply callback if conditions are met.

        Note that the current result may be modified by specifying a function
        `modify_optimize_result` for instance to add inputs to the `extra` argument
        of the current OptimizeResult.

        Args:
            opt_result (OptimizeResult): Current result.
            is_last_iteration (bool, optional): When True,
                avoid verifying modulo. Defaults to False.
                Useful when call_after_opt is True.

        Returns:
            Any: The result of the callback.
        """
        opt_result = self.modify_optimize_result(opt_result)
        if opt_result.iteration % self.called_every == 0 and self.callback_condition(opt_result):
            return self.callback(opt_result)
        if is_last_iteration and self.callback_condition(opt_result):
            return self.callback(opt_result)


def run_callbacks(
    callback_iterable: list[Callback], opt_res: OptimizeResult, is_last_iteration: bool = False
) -> None:
    """Run a list of Callback given the current OptimizeResult.

    Used in train functions.

    Args:
        callback_iterable (list[Callback]): Iterable of Callbacks
        opt_res (OptimizeResult): Current optimization result,
        is_last_iteration (bool, optional): Whether we reached the last iteration or not.
            Defaults to False.
    """
    for callback in callback_iterable:
        callback(opt_res, is_last_iteration)


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
    """Print loss/metrics.

    Set to 0 to disable
    """
    write_every: int = 50
    """Write loss and metrics with the tracking tool.

    Set to 0 to disable
    """
    checkpoint_every: int = 5000
    """Write model/optimizer checkpoint.

    Set to 0 to disable
    """
    plot_every: int = 5000
    """Write figures.

    Set to 0 to disable
    """
    callbacks: list[Callback] = field(default_factory=lambda: list())
    """List of callbacks."""
    log_model: bool = False
    """Logs a serialised version of the model."""
    folder: Path | None = None
    """Checkpoint/tensorboard logs folder."""
    create_subfolder_per_run: bool = False
    """Checkpoint/tensorboard logs stored in subfolder with name `<timestamp>_<PID>`.

    Prevents continuing from previous checkpoint, useful for fast prototyping.
    """
    checkpoint_best_only: bool = False
    """Write model/optimizer checkpoint only if a metric has improved."""
    val_every: int | None = None
    """Calculate validation metric.

    If None, validation check is not performed.
    """
    val_epsilon: float = 1e-5
    """Safety margin to check if validation loss is smaller than the lowest.

    validation loss across previous iterations.
    """
    validation_criterion: Callable | None = None
    """A boolean function which evaluates a given validation metric is satisfied."""
    trainstop_criterion: Callable | None = None
    """A boolean function which evaluates a given training stopping metric is satisfied."""
    batch_size: int = 1
    """The batch_size to use when passing a list/tuple of torch.Tensors."""
    verbose: bool = True
    """Whether or not to print out metrics values during training."""
    tracking_tool: ExperimentTrackingTool = ExperimentTrackingTool.TENSORBOARD
    """The tracking tool of choice."""
    hyperparams: dict = field(default_factory=dict)
    """Hyperparameters to track."""
    plotting_functions: tuple[LoggablePlotFunction, ...] = field(default_factory=tuple)  # type: ignore
    """Functions for in-train plotting."""

    # tensorboard only allows for certain types as hyperparameters
    _tb_allowed_hyperparams_types: tuple = field(
        default=(int, float, str, bool, Tensor), init=False, repr=False
    )

    def _filter_tb_hyperparams(self) -> None:
        keys_to_remove = [
            key
            for key, value in self.hyperparams.items()
            if not isinstance(value, TrainConfig._tb_allowed_hyperparams_types)
        ]
        if keys_to_remove:
            logger.warning(
                f"Tensorboard cannot log the following hyperparameters: {keys_to_remove}."
            )
            for key in keys_to_remove:
                self.hyperparams.pop(key)

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
            self.validation_criterion = lambda *x: False
        if self.hyperparams and self.tracking_tool == ExperimentTrackingTool.TENSORBOARD:
            self._filter_tb_hyperparams()
        if self.tracking_tool == ExperimentTrackingTool.MLFLOW:
            self._mlflow_config = MLFlowConfig()
        if self.plotting_functions and self.tracking_tool != ExperimentTrackingTool.MLFLOW:
            logger.warning("In-training plots are only available with mlflow tracking.")
        if not self.plotting_functions and self.tracking_tool == ExperimentTrackingTool.MLFLOW:
            logger.warning("Tracking with mlflow, but no plotting functions provided.")

    @property
    def mlflow_config(self) -> MLFlowConfig:
        if self.tracking_tool == ExperimentTrackingTool.MLFLOW:
            return self._mlflow_config
        else:
            raise AttributeError(
                "mlflow_config is available only for with the mlflow tracking tool."
            )


class MLFlowConfig:
    """
    Configuration for mlflow tracking.

    Example:

        export MLFLOW_TRACKING_URI=tracking_uri
        export MLFLOW_EXPERIMENT=experiment_name
        export MLFLOW_RUN_NAME=run_name
    """

    def __init__(self) -> None:
        import mlflow

        self.tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
        """The URI of the mlflow tracking server.

        An empty string, or a local file path, prefixed with file:/.
        Data is stored locally at the provided file (or ./mlruns if empty).
        """

        self.experiment_name: str = os.getenv("MLFLOW_EXPERIMENT", str(uuid4()))
        """The name of the experiment.

        If None or empty, a new experiment is created with a random UUID.
        """

        self.run_name: str = os.getenv("MLFLOW_RUN_NAME", str(uuid4()))
        """The name of the run."""

        mlflow.set_tracking_uri(self.tracking_uri)

        # activate existing or create experiment
        exp_filter_string = f"name = '{self.experiment_name}'"
        if not mlflow.search_experiments(filter_string=exp_filter_string):
            mlflow.create_experiment(name=self.experiment_name)

        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name, nested=False)


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
    `AnsatzType.IIA` for Identity intialized Ansatz.
    """

    ansatz_strategy: Strategy = Strategy.DIGITAL
    """Ansatz strategy.

    `Strategy.DIGITAL` for fully digital ansatz. Required if `ansatz_type` is `AnsatzType.IIA`.
    `Strategy.SDAQC` for analog entangling block.
    `Strategy.RYDBERG` for fully rydberg hea ansatz.
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
