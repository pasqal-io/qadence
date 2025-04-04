from __future__ import annotations

from typing import Any, Callable
from qadence.blocks import chain
from qadence.parameters import Parameter
from qadence.blocks.utils import add, tag
from qadence.operations import Z, RX, CZ
from qadence.circuit import QuantumCircuit
from qadence.types import BackendName, DiffMode
from qadence.blocks.abstract import AbstractBlock

from .models import QNN
from qadence.ml_tools.constructors import _create_conv_layer, _create_feature_map_qcnn


class QCNN(QNN):
    def __init__(
        self,
        n_inputs: int,
        n_qubits: int,
        depth: list[int],
        operations: list[Any],
        entangler: Any = CZ,
        random_meas: bool = True,
        fm_basis: str = "Fourier",
        fm_gate: Any = RX,
        is_corr: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Creates a QCNN model.

        Args:
            n_inputs (int): Number of input features.
            n_qubits (int): Total number of qubits.
            depth (list[int]): List defining the depth (repetitions) of each layer.
            operations (list[Any]): List of quantum operations to apply
                in the gates (e.g., [RX, RZ]).
            entangler (Any): Entangling operation, such as CZ.
            random_meas (bool): If True, applies random weighted measurements.
            fm_basis (str): feature map basis.
            fm_gate (Any): gate employed in the fm, such as.
            **kwargs (Any): Additional keyword arguments for the parent QNN class.
        """
        self.n_inputs = n_inputs
        self.n_qubits = n_qubits
        self.depth = depth
        self.operations = operations
        self.entangler = entangler
        self.random_meas = random_meas
        self.fm_basis = fm_basis
        self.fm_gate = fm_gate
        self.is_corr = is_corr

        circuit = self.qcnn_circuit(
            self.n_inputs,
            self.n_qubits,
            self.depth,
            self.operations,
            self.entangler,
            self.fm_basis,
            self.fm_gate,
            self.is_corr,
        )

        obs = self.qcnn_deferred_obs(self.n_qubits, self.random_meas)

        super().__init__(
            circuit=circuit,
            observable=obs,
            backend=BackendName.PYQTORCH,
            diff_mode=DiffMode.AD,
            inputs=[f"\u03C6_{i}" for i in range(self.n_inputs)],
            **kwargs,
        )

    def qcnn_circuit(
        self,
        n_inputs: int,
        n_qubits: int,
        depth: list[int],
        operations: list[Any],
        entangler: AbstractBlock,
        fm_basis: str,
        fm_gate: AbstractBlock,
        is_corr: bool,
    ) -> QuantumCircuit:
        """Defines the QCNN circuit."""
        # Validate qubit count
        if n_qubits < 4:
            raise ValueError(
                f"Invalid number of qubits: {n_qubits}. " "At least 4 qubits are required."
            )
        if n_qubits % 2 != 0:
            raise ValueError(
                f"Invalid number of qubits: {n_qubits}. " "The number of qubits must be even."
            )

        # Validate that all values in `depth` are odd
        even_depths = [d for d in depth if d % 2 == 0]
        if even_depths:
            raise ValueError(
                f"Invalid depth values: '{even_depths[0]}'. " "All the conv layer 'r's must be odd."
            )

        # Feature map (FM)
        fm = _create_feature_map_qcnn(n_qubits, n_inputs, fm_basis, fm_gate)
        tag(fm, "FM")

        # Conv and Pool layer definition
        conv_layers = []
        params: dict[str, Parameter] = {}

        # Define layer all the 2-qubit patterns based on depth
        layer_patterns = [(2**layer_index, depth[layer_index]) for layer_index in range(len(depth))]

        # Initialize all qubits for the current layer
        current_indices = list(range(n_qubits))

        # Build the circuit layer by layer using the helper
        for layer_index, (_, reps) in enumerate(layer_patterns):
            if reps == 0:
                raise ValueError(f"Invalid layer {layer_index}: zero repetitions (reps = {reps}).")
            if len(current_indices) < 2:
                raise RuntimeError(
                    f"Layer {layer_index} requires at least 2 qubits, "
                    f"but found {len(current_indices)}."
                )

            layer_block, next_indices = _create_conv_layer(
                layer_index, reps, current_indices, params, operations, entangler, n_qubits, is_corr
            )
            tag(layer_block, f"C+P layer {layer_index}")
            conv_layers.append(layer_block)

            # Update `current_indices` for the next layer
            current_indices = next_indices

        # Combine all layers for the final ansatz
        ansatz = chain(*conv_layers)

        return QuantumCircuit(n_qubits, fm, ansatz)

    def qcnn_deferred_obs(
        self, n_qubits: int, random_meas: bool
    ) -> AbstractBlock | list[AbstractBlock]:
        """
        Defines the measurements to be performedthe traced out.

        and remaining qubits.
        """
        if random_meas:
            w1 = [Parameter(f"w{i}") for i in range(n_qubits)]
            obs = add(Z(i) * w for i, w in zip(range(n_qubits), w1))
        else:
            obs = add(Z(i) for i in range(n_qubits))

        return obs
