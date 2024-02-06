from __future__ import annotations

from collections import Counter
from typing import Callable

import sympy
from torch import Tensor, nn

from qadence.backend import BackendConfiguration, ConvertedObservable
from qadence.backends.api import config_factory
from qadence.blocks.abstract import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.logger import get_logger
from qadence.measurements import Measurements
from qadence.mitigations import Mitigations
from qadence.models.quantum_model import QuantumModel
from qadence.noise import Noise
from qadence.types import BackendName, DiffMode, Endianness

logger = get_logger(__name__)


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
        transform: Callable[[Tensor], Tensor] = None,  # transform output of the QNN
        backend: BackendName = BackendName.PYQTORCH,
        diff_mode: DiffMode = DiffMode.AD,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
        configuration: BackendConfiguration | dict | None = None,
        inputs: list[sympy.Basic | str] | None = None,
    ):
        """Initialize the QNN.

        The number of inputs is determined by the feature parameters in the input
        quantum circuit while the number of outputs is determined by how many
        observables are provided as input

        Args:
            circuit: The quantum circuit to use for the QNN.
            transform: A transformation applied to the output of the QNN.
            inputs: Tuple that indicates the order of variables of the tensors that are passed
                to the model. Given input tensors `xs = torch.rand(batch_size, input_size:=2)` a QNN
                with `inputs=("t", "x")` will assign `t, x = xs[:,0], xs[:,1]`.
            backend: The chosen quantum backend.
            diff_mode: The differentiation engine to use. Choices 'gpsr' or 'ad'.
            measurement: optional measurement protocol. If None,
                use exact expectation value with a statevector simulator
            noise: A noise model to use.
            configuration: optional configuration for the backend
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
        if self.out_features is None:
            raise ValueError("You need to provide at least one observable in the QNN constructor")
        self.transform = transform if transform else lambda x: x

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

    def forward(
        self,
        values: dict[str, Tensor] | Tensor = None,
        state: Tensor | None = None,
        measurement: Measurements | None = None,
        noise: Noise | None = None,
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
        return super().run(values=self._format_to_dict(values), state=state, endianness=endianness)

    def sample(
        self,
        values: Tensor | dict[str, Tensor] = {},
        n_shots: int = 1000,
        state: Tensor | None = None,
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> list[Counter]:
        return super().sample(
            values=self._format_to_dict(values),
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
        noise: Noise | None = None,
        mitigation: Mitigations | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> Tensor:
        if values is None:
            values = {}
        if measurement is None:
            measurement = self._measurement
        if noise is None:
            noise = self._noise
        return self.transform(
            super().expectation(
                values=self._format_to_dict(values),
                state=state,
                measurement=measurement,
                endianness=endianness,
                noise=noise,
            )
        )

    def _format_to_dict(self, values: Tensor) -> dict[str, Tensor]:
        """Format an input tensor into the format required by the forward pass.

        The tensor is assumed to have dimensions: n_batches x in_features where in_features
        corresponds to the number of input features of the QNN
        """
        # for backwards compat...
        if isinstance(values, dict):
            return values

        if len(values.size()) == 1:
            values = values.reshape(-1, 1)
        msg = f"Model expects in_features={self.in_features} but got {values.size()[1]}."
        assert len(values.size()) == 2, msg
        assert values.size()[1] == self.in_features, msg

        return {var.name: values[:, self.inputs.index(var)] for var in self.inputs}

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
                noise=Noise._from_dict(qm_dict["noise"]),
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
