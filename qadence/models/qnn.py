from __future__ import annotations

from typing import Callable

from torch import Tensor

from qadence.backend import BackendConfiguration
from qadence.blocks.abstract import AbstractBlock
from qadence.circuit import QuantumCircuit
from qadence.measurements import Measurements
from qadence.models.quantum_model import QuantumModel
from qadence.noise import Noise
from qadence.types import BackendName, DiffMode, Endianness


class QNN(QuantumModel):
    """Quantum neural network model for n-dimensional inputs.

    Examples:
    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import QuantumCircuit, QNN
    from qadence import hea, feature_map, hamiltonian_factory, Z

    # create the circuit
    n_qubits, depth = 2, 4
    fm = feature_map(n_qubits)
    ansatz = hea(n_qubits=n_qubits, depth=depth)
    circuit = QuantumCircuit(n_qubits, fm, ansatz)
    obs_base = hamiltonian_factory(n_qubits, detuning = Z)

    # the QNN will yield two outputs
    obs = [2.0 * obs_base, 4.0 * obs_base]

    # initialize and use the model
    qnn = QNN(circuit, obs, diff_mode="ad", backend="pyqtorch")
    y = qnn.expectation({"phi": torch.rand(3)})
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
    ):
        """Initialize the QNN.

        The number of inputs is determined by the feature parameters in the input
        quantum circuit while the number of outputs is determined by how many
        observables are provided as input

        Args:
            circuit: The quantum circuit to use for the QNN.
            transform: A transformation applied to the output of the QNN.
            backend: The chosen quantum backend.
            diff_mode: The differentiation engine to use. Choices 'gpsr' or 'ad'.
            measurement: optional measurement protocol. If None,
                use exact expectation value with a statevector simulator
            noise: A noise model to use.
            configuration: optional configuration for the backend
        """
        super().__init__(
            circuit=circuit,
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
            values (dict[str, Tensor] | Tensor): the values of the feature parameters
            state: Initial state.
            measurement: optional measurement protocol. If None,
                use exact expectation value with a statevector simulator
            noise: A noise model to use.
            endianness: Endianness of the resulting bit strings.

        Returns:
            Tensor: a tensor with the expectation value of the observables passed
                in the constructor of the model
        """
        if values is None:
            values = {}
        if not isinstance(values, dict):
            values = self._format_to_dict(values)
        if measurement is None:
            measurement = self._measurement
        if noise is None:
            noise = self._noise

        return self.transform(
            self.expectation(
                values=values,
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

        if len(values.size()) == 1:
            values = values.reshape(-1, 1)
        msg = f"Model expects in_features={self.in_features} but got {values.size()[1]}."
        assert len(values.size()) == 2, msg
        assert values.size()[1] == self.in_features, msg

        names = [p.name for p in self.inputs]
        res = {}
        for i, name in enumerate(names):
            res[name] = values[:, i]
        return res

    # TODO: Implement derivatives w.r.t. to inputs
