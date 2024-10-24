from __future__ import annotations

from collections import Counter
from logging import getLogger
from typing import Any, Callable

import sympy
import torch
from torch import Tensor, nn

from qadence.backend import BackendConfiguration, ConvertedObservable
from qadence.backends.api import config_factory
from qadence.blocks.abstract import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.ml_tools.config import AnsatzConfig, FeatureMapConfig
from qadence.model import QuantumModel
from qadence.noise import NoiseHandler
from qadence.register import Register
from qadence.types import BackendName, DiffMode, Endianness, InputDiffMode, ParamDictType

logger = getLogger(__name__)


def _torch_derivative(
    ufa: Callable, x: torch.Tensor, derivative_indices: tuple[int, ...]
) -> torch.Tensor:
    y = ufa(x)
    for idx in derivative_indices:
        out = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        y = out[:, idx]
    return y.reshape(-1, 1)


def derivative(ufa: torch.nn.Module, x: Tensor, derivative_indices: tuple[int, ...]) -> Tensor:
    """Compute derivatives w.r.t.

    inputs of a UFA with a single output. The
    `derivative_indices` specify which derivative(s) are computed.  E.g.
    `derivative_indices=(1,2)` would compute the a second order derivative w.r.t
    to the indices `1` and `2` of the input tensor.

    Arguments:
        ufa: The model for which we want to compute the derivative.
        x (Tensor): (batch_size, input_size) input tensor.
        derivative_indices (tuple): Define which derivatives to compute.

    Examples:
    If we create a UFA with three inputs and denote the first, second, and third
    input with `x`, `y`, and `z` we can compute the following derivatives w.r.t
    to those inputs:
    ```py exec="on" source="material-block"
    import torch
    from qadence.ml_tools.models import derivative, QNN
    from qadence.ml_tools.config import FeatureMapConfig, AnsatzConfig
    from qadence.constructors.hamiltonians import ObservableConfig
    from qadence.operations import Z

    fm_config = FeatureMapConfig(num_features=3, inputs=["x", "y", "z"])
    ansatz_config = AnsatzConfig()
    obs_config = ObservableConfig(detuning=Z)

    f = QNN.from_configs(
        register=3, obs_config=obs_config, fm_config=fm_config, ansatz_config=ansatz_config,
    )
    inputs = torch.rand(5,3,requires_grad=True)

    # df_dx
    derivative(f, inputs, (0,))

    # d2f_dydz
    derivative(f, inputs, (1,2))

    # d3fdy2dx
    derivative(f, inputs, (1,1,0))
    ```
    """
    assert ufa.out_features == 1, "Can only call `derivative` on models with 1D output."
    return ufa._derivative(x, derivative_indices)


def format_to_dict_fn(
    inputs: list[sympy.Symbol | str] = [],
) -> Callable[[Tensor | ParamDictType], ParamDictType]:
    """Format an input tensor into the format required by the forward pass.

    The tensor is assumed to have dimensions: n_batches x in_features where in_features
    corresponds to the number of input features of the QNN
    """
    in_features = len(inputs)

    def tensor_to_dict(values: Tensor | ParamDictType) -> ParamDictType:
        if isinstance(values, Tensor):
            values = values.reshape(-1, 1) if len(values.size()) == 1 else values
            if not values.shape[1] == in_features:
                raise ValueError(
                    f"Model expects in_features={in_features} but got {values.shape[1]}."
                )
            values = {fparam.name: values[:, inputs.index(fparam)] for fparam in inputs}  # type: ignore[union-attr]
        return values

    return tensor_to_dict


class QNN(QuantumModel):
    """Quantum neural network model for n-dimensional inputs.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import QuantumCircuit, QNN, Z
    from qadence import hea, feature_map, hamiltonian_factory, kron

    # create the circuit
    n_qubits, depth = 2, 4
    fm = kron(
        feature_map(1, support=(0,), param="x"),
        feature_map(1, support=(1,), param="y")
    )
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs_base = hamiltonian_factory(n_qubits, detuning=Z)

    # the QNN will yield two outputs
    obs = [2.0 * obs_base, 4.0 * obs_base]

    # initialize and use the model
    qnn = QNN(circuit, obs, inputs=["x", "y"])
    y = qnn(torch.rand(3, 2))
    print(str(y)) # markdown-exec: hide
    ```
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: list[AbstractBlock] | AbstractBlock,
        backend: BackendName = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        configuration: BackendConfiguration | dict | None = None,
        inputs: list[sympy.Basic | str] | None = None,
        input_diff_mode: InputDiffMode | str = InputDiffMode.AD,
    ):
        """Initialize the QNN.

        The number of inputs is determined by the feature parameters in the input
        quantum circuit while the number of outputs is determined by how many
        observables are provided as input

        Args:
            circuit: The quantum circuit to use for the QNN.
            observable: The observable.
            backend: The chosen quantum backend.
            diff_mode: The differentiation engine to use. Choices 'gpsr' or 'ad'.
            measurement: optional measurement protocol. If None,
                use exact expectation value with a statevector simulator
            noise: A noise model to use.
            configuration: optional configuration for the backend
            inputs: List that indicates the order of variables of the tensors that are passed
                to the model. Given input tensors `xs = torch.rand(batch_size, input_size:=2)` a QNN
                with `inputs=["t", "x"]` will assign `t, x = xs[:,0], xs[:,1]`.
            input_diff_mode: The differentiation mode for the input tensor.
        """
        super().__init__(
            circuit,
            observable=observable,
            backend=backend,
            diff_mode=diff_mode,
            measurement=measurement,
            configuration=configuration,
            noise=noise,
        )
        if self._observable is None:
            raise ValueError("You need to provide at least one observable in the QNN constructor")
        if (inputs is not None) and (len(self.inputs) == len(inputs)):
            self.inputs = [sympy.symbols(x) if isinstance(x, str) else x for x in inputs]  # type: ignore[union-attr]
        elif (inputs is None) and len(self.inputs) <= 1:
            self.inputs = [sympy.symbols(x) if isinstance(x, str) else x for x in self.inputs]  # type: ignore[union-attr]
        else:
            raise ValueError(
                """
                Your QNN has more than one input. Please provide a list of inputs in the order of
                your tensor domain. For example, if you want to pass
                `xs = torch.rand(batch_size, input_size:=3)` to you QNN, where
                ```
                t = x[:,0]
                x = x[:,1]
                y = x[:,2]
                ```
                you have to specify
                ```
                QNN(circuit, observable, inputs=["t", "x", "y"])
                ```
                You can also pass a list of sympy symbols.
            """
            )
        self.format_to_dict = format_to_dict_fn(self.inputs)  # type: ignore[arg-type]
        self.input_diff_mode = InputDiffMode(input_diff_mode)
        if self.input_diff_mode == InputDiffMode.FD:
            from qadence.backends.utils import finitediff

            self.__derivative = finitediff
        elif self.input_diff_mode == InputDiffMode.AD:
            self.__derivative = _torch_derivative  # type: ignore[assignment]
        else:
            raise ValueError(f"Unkown forward diff mode: {self.input_diff_mode}")

    @classmethod
    def from_configs(
        cls,
        register: int | Register,
        obs_config: Any,
        fm_config: Any = FeatureMapConfig(),
        ansatz_config: Any = AnsatzConfig(),
        backend: BackendName = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        configuration: BackendConfiguration | dict | None = None,
        input_diff_mode: InputDiffMode | str = InputDiffMode.AD,
    ) -> QNN:
        """Create a QNN from a set of configurations.

        Args:
            register (int | Register): The number of qubits or a register object.
            obs_config (list[ObservableConfig] | ObservableConfig): The configuration(s)
                for the observable(s).
            fm_config (FeatureMapConfig): The configuration for the feature map.
                Defaults to no feature encoding block.
            ansatz_config (AnsatzConfig): The configuration for the ansatz.
                Defaults to a single layer of hardware efficient ansatz.
            backend (BackendName): The chosen quantum backend.
            diff_mode (DiffMode): The differentiation engine to use. Choices are
                'gpsr' or 'ad'.
            measurement (Measurements): Optional measurement protocol. If None,
                use exact expectation value with a statevector simulator.
            noise (Noise): A noise model to use.
            configuration (BackendConfiguration | dict): Optional backend configuration.
            input_diff_mode (InputDiffMode): The differentiation mode for the input tensor.

        Returns:
            A QNN object.

        Raises:
            ValueError: If the observable configuration is not provided.

        Example:
        ```python exec="on" source="material-block" result="json"
        import torch
        from qadence.ml_tools.config import AnsatzConfig, FeatureMapConfig
        from qadence.ml_tools import QNN
        from qadence.constructors import ObservableConfig
        from qadence.operations import Z
        from qadence.types import (
            AnsatzType, BackendName, BasisSet, ObservableTransform, ReuploadScaling, Strategy
        )

        register = 4
        obs_config = ObservableConfig(
            detuning=Z,
            scale=5.0,
            shift=0.0,
            transformation_type=ObservableTransform.SCALE,
            trainable_transform=None,
        )
        fm_config = FeatureMapConfig(
            num_features=2,
            inputs=["x", "y"],
            basis_set=BasisSet.FOURIER,
            reupload_scaling=ReuploadScaling.CONSTANT,
            feature_range={
                "x": (-1.0, 1.0),
                "y": (0.0, 1.0),
            },
        )
        ansatz_config = AnsatzConfig(
            depth=2,
            ansatz_type=AnsatzType.HEA,
            ansatz_strategy=Strategy.DIGITAL,
        )

        qnn = QNN.from_configs(
            register, obs_config, fm_config, ansatz_config, backend=BackendName.PYQTORCH
        )

        x = torch.rand(2, 2)
        y = qnn(x)
        print(str(y)) # markdown-exec: hide
        ```
        """
        from .constructors import build_qnn_from_configs

        return build_qnn_from_configs(
            register=register,
            observable_config=obs_config,
            fm_config=fm_config,
            ansatz_config=ansatz_config,
            backend=backend,
            diff_mode=diff_mode,
            measurement=measurement,
            noise=noise,
            configuration=configuration,
            input_diff_mode=input_diff_mode,
        )

    def forward(
        self,
        values: dict[str, Tensor] | Tensor = None,
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        """Forward pass of the model.

        This returns the (differentiable) expectation value of the given observable
        operator defined in the constructor. Differently from the base QuantumModel
        class, the QNN accepts also a tensor as input for the forward pass. The
        tensor is expected to have shape: `n_batches x in_features` where `n_batches`
        is the number of data points and `in_features` is the dimensionality of the problem

        The output of the forward pass is the expectation value of the input
        observable(s). If a single observable is given, the output shape is
        `n_batches` while if multiple observables are given the output shape
        is instead `n_batches x n_observables`

        Args:
            values: the values of the feature parameters
            state: Initial state.
            measurement: optional measurement protocol. If None,
                use exact expectation value with a statevector simulator
            noise: A noise model to use.
            endianness: Endianness of the resulting bit strings.

        Returns:
            Tensor: a tensor with the expectation value of the observables passed
                in the constructor of the model
        """
        return self.expectation(
            values, state=state, measurement=measurement, noise=noise, endianness=endianness
        )

    def run(
        self,
        values: Tensor | dict[str, Tensor] = None,
        state: Tensor | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        return super().run(
            values=self.format_to_dict(values),
            state=state,
            endianness=endianness,
        )

    def sample(
        self,
        values: Tensor | dict[str, Tensor] = {},
        n_shots: int = 1000,
        state: Tensor | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        return super().sample(
            values=self.format_to_dict(values),
            n_shots=n_shots,
            state=state,
            noise=noise,
            mitigation=mitigation,
            endianness=endianness,
        )

    def expectation(
        self,
        values: Tensor | dict[str, Tensor] = {},
        observable: list[ConvertedObservable] | ConvertedObservable | None = None,
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: NoiseHandler | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        if values is None:
            values = {}
        if measurement is None:
            measurement = self._measurement
        if noise is None:
            noise = self._noise
        return super().expectation(
            values=self.format_to_dict(values),
            state=state,
            measurement=measurement,
            endianness=endianness,
            noise=noise,
        )

    def _derivative(self, x: Tensor, derivative_indices: tuple[int, ...]) -> Tensor:
        return self.__derivative(self, x, derivative_indices)

    def _to_dict(self, save_params: bool = False) -> dict:
        d = dict()
        try:
            d = super()._to_dict(save_params)
            d[self.__class__.__name__]["inputs"] = [str(i) for i in self.inputs]
            logger.debug(f"{self.__class__.__name__} serialized to {d}.")
        except Exception as e:
            logger.warning(f"Unable to serialize {self.__class__.__name__} due to {e}.")
        return d

    @classmethod
    def _from_dict(cls, d: dict, as_torch: bool = False) -> QNN:
        from qadence.serialization import deserialize

        qnn: QNN
        try:
            qm_dict = d[cls.__name__]
            qnn = cls(
                circuit=QuantumCircuit._from_dict(qm_dict["circuit"]),
                observable=[deserialize(q_obs) for q_obs in qm_dict["observable"]],  # type: ignore[misc]
                backend=qm_dict["backend"],
                diff_mode=qm_dict["diff_mode"],
                measurement=Measurements._from_dict(qm_dict["measurement"]),
                noise=NoiseHandler._from_dict(qm_dict["noise"]),
                configuration=config_factory(qm_dict["backend"], qm_dict["backend_configuration"]),
                inputs=qm_dict["inputs"],
            )

            if as_torch:
                conv_pd = nn.ParameterDict()
                param_dict = d["param_dict"]
                for n, param in param_dict.items():
                    conv_pd[n] = nn.Parameter(param)
                qnn._params = conv_pd
            logger.debug(f"Initialized {cls.__name__} from {d}.")

        except Exception as e:
            logger.warning(f"Unable to deserialize object {d} to {cls.__name__} due to {e}.")

        return qnn
