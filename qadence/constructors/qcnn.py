from __future__ import annotations
from typing import Any
import torch

from qadence.constructors import feature_map
from qadence.ml_tools.models import QNN
from qadence.blocks import chain, kron
from qadence.blocks.utils import add, tag
from qadence.blocks.abstract import AbstractBlock
from qadence.operations import RX, Z
from qadence.circuit import QuantumCircuit
from qadence.types import BackendName, DiffMode
from qadence.parameters import Parameter

from qadence.ml_tools.constructors import __create_layer

####
class qcnn(QNN):
    def __init__(
        self,
        n_inputs: int,
        n_qubits: int,
        depth: list[int],
        operations: list[Any],
        entangler: Any,
        random_meas: bool,
        use_dagger: bool,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the qcnn class.

        Args:
            n_inputs (int): Number of input features.
            n_qubits (int): Total number of qubits.
            depth (list[int]): List defining the depth (repetitions) of each layer.
            operations (list[Any]): List of quantum operations to apply in the gates
                (e.g., [RX, RZ]).
            entangler (Any): Entangling operation, such as CNOT or CZ.
            random_meas (bool): If True, applies random weighted measurements.
            use_dagger (bool): If True, includes the corresponding adjoint operations
                (gates with negative angles).
            **kwargs (Any): Additional keyword arguments for the parent QNN class.
        """
        self.n_inputs = n_inputs
        self.n_qubits = n_qubits
        self.depth = depth
        self.operations = operations
        self.entangler = entangler
        self.random_meas = random_meas

        # Create circuit and observables
        circuit, _ = self.create_circuit(
            self.n_inputs, self.n_qubits, self.depth, self.operations, self.entangler, use_dagger
        )
        obs = self.create_obs(self.n_qubits, self.random_meas)

        super().__init__(
            circuit=circuit,
            observable=obs,
            backend=BackendName.PYQTORCH,
            diff_mode=DiffMode.AD,
            inputs=[f"\u03C6_{i}" for i in range(self.n_inputs)],
            **kwargs,
        )

    def create_circuit(
        self, n_inputs: int, n_qubits: int, depth: list[int], operations: list[Any], entangler:Any, use_dagger:bool,
    ) -> tuple[QuantumCircuit, list[int]]:
        '''
        Defines a single, continuous quantum circuit with custom repeating ansatz for each depth.

        Args:
            n_inputs (int): Number of input features.
            n_qubits (int): Total number of qubits.
            depth (list[int]): List defining the depth (repetitions) of each layer.
            operations (list[Any]): List of quantum operations to apply in the gates.
            entangler (Any): Entangling operation, such as CNOT or CZ.

        Returns:
            tuple[QuantumCircuit, list[int]]: A tuple containing the quantum circuit and the final target indices.
        '''
        # Feature map (FM)
        fm_temp = [
            feature_map(
                n_qubits=(n_qubits // n_inputs),
                param=f"\u03C6_{i}",
                op=RX,
                fm_type="Fourier",
                support=tuple(range(i * (n_qubits // n_inputs), (i + 1) * (n_qubits // n_inputs))),
            )
            for i in range(n_inputs)
        ]
        fm = kron(*fm_temp)
        tag(fm, "FM")

        ansatz_layers = []  # To store each layer of the ansatz
        params: dict[str, Parameter] = {}
        all_target_indices = []  # To store target indices for each layer

        # Define layer patterns based on depth
        layer_patterns = [(2 ** layer_index, depth[layer_index]) for layer_index in range(len(depth))]

        # Initialize all qubits for the first layer
        current_indices = list(range(n_qubits))

        # Build the circuit layer by layer using the helper
        for layer_index, (_, reps) in enumerate(layer_patterns):
            if reps == 0 or len(current_indices) < 2:
                break  # Skip this layer if depth is 0 or fewer than 2 qubits remain

            layer_block, next_indices = __create_layer(
                layer_index, reps, current_indices, params, operations, entangler, n_qubits, use_dagger
            )

            # Append the current `current_indices` to `all_target_indices`
            all_target_indices.append(current_indices)

            # Update `current_indices` for the next layer
            current_indices = next_indices

            tag(layer_block, f"Layer {layer_index}")
            ansatz_layers.append(layer_block)

        # Combine all layers for the final ansatz
        ansatz = chain(*ansatz_layers)
        tag(ansatz, "Ansatz")

        qc = QuantumCircuit(n_qubits, fm, ansatz)
        return qc, next_indices
    
    def create_obs(self, n_qubits: int, random_meas: bool) -> AbstractBlock | list[AbstractBlock]:
        """
        Defines the measurements to be performed on the specified target qubits.

        Args:
            n_qubits (int): Total number of qubits.
            random_meas (bool): If True, applies random weighted measurements.
            last_target_qubits (list[int]): List of target qubits to measure.

        Returns:
            AbstractBlock | list[AbstractBlock]: The measurement observable
                or a list of observables.
        """
        if random_meas:
            w1 = [Parameter(f"w{i}") for i in range(n_qubits)]
            obs = add(Z(i) * w for i, w in zip(range(n_qubits), w1))
        else:
            obs = add(Z(i) for i in range(n_qubits))

        print(obs)
        return obs


class qcnn_msg_passing(torch.nn.Module):
    """
    QCNN message-passing module.

    Processes input features sequentially through a list of QNN models and
    returns the mean of the final model's output.

    Example:
        qcnn_list = torch.nn.ModuleList([QCNN(**kwargs1),QCNN(**kwargs2),QCNN(**kwargs3)])
        combined_model = qcnn_msg_passing(qcnn_list, qgcn_output_size=4)

    Args:
        qgcn_list (torch.nn.ModuleList): List of QNN models.
        qgcn_output_size (int): Size of the output features for each model.

    Returns:
        torch.Tensor: Mean of the final QNN model's output.
    """

    def __init__(self, qgcn_list: torch.nn.ModuleList, qgcn_output_size: int = 1) -> None:
        """
        Initialize the QCNN message-passing module.

        Args:
            qgcn_list (torch.nn.ModuleList): List of QNN models.
            qgcn_output_size (int): Size of the output features for each model.
        """
        super(qcnn_msg_passing, self).__init__()
        self.qgcn_list: torch.nn.ModuleList = torch.nn.ModuleList(qgcn_list)
        self.qgcn_output_size: int = qgcn_output_size

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QCNN message-passing network.

        Args:
            combined_features (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Mean of the final QNN model's output.
        """
        x = combined_features

        for qgcn in self.qgcn_list:
            qgcn_output = qgcn(x).float()
            x = qgcn_output.view(-1, self.qgcn_output_size).float()

        return torch.mean(qgcn_output, dim=0, keepdim=True)
