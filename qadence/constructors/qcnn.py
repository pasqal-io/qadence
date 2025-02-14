class qcnn(QNN):
    def __init__(
        self,
        n_inputs: int,
        n_qubits: int,
        depth: list[int],
        operations: list[Any],
        entangler: Any,
        random_meas: bool,
        use_dagger: bool = True,  # True for using the adjoint operation
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
        self.use_dagger = use_dagger  # Store the flag

        # Create circuit and observables
        circuit, last_target_qubits = self.create_circuit(
            self.n_inputs, self.n_qubits, self.depth, self.operations, self.entangler
        )
        obs = self.create_obs(self.n_qubits, self.random_meas, last_target_qubits)

        super().__init__(
            circuit=circuit,
            observable=obs,
            backend=BackendName.PYQTORCH,
            diff_mode=DiffMode.AD,
            inputs=[f"\u03C6_{i}" for i in range(self.n_inputs)],
            **kwargs,
        )

    def get_param(self, params: dict, layer: int, rep: int, pos: int) -> Parameter:
        """
        Retrieves or creates a parameter key for the given layer, repetition, and position.

        Args:
            params (dict): Dictionary to store and retrieve parameters.
            layer (int): The index of the current layer.
            rep (int): The index of the current repetition in the layer.
            pos (int): Position of the qubit in the layer.

        Returns:
            Parameter: The created or retrieved parameter.
        """
        key = f"\u03B8_{layer}_{rep}_{pos}"
        if key not in params:
            params[key] = Parameter(key)
        return params[key]

    def create_gate_sequence(
        self,
        params: dict,
        operations: list[Any],
        entangler: Any,
        layer: int,
        rep: int,
        control: int,
        target: int,
        spacing: int = 0,
        n_qubits: int = 8,
    ) -> AbstractBlock:
        """
        Creates a sequence of gates for a control-target pair.

        Args:
            params (dict): Dictionary to store and retrieve parameters.
            operations (list[Any]): List of gate operations to apply (e.g., [RX, RZ]).
            entangler (Any): Entangling operation, such as CNOT or CZ.
            layer (int): The index of the current layer.
            rep (int): The index of the current repetition in the layer.
            control (int): Index of the control qubit.
            target (int): Index of the target qubit.
            spacing (int, optional): Number of qubits to include as padding. Defaults to 0.
            n_qubits (int, optional): Total number of qubits. Defaults to 8.

        Returns:
            AbstractBlock: The sequence of gates as a quantum block.
        """
        pad = [
            I(q)
            for q in range(control - spacing, control + spacing + 1)
            if q != control and q != target and 0 <= q < n_qubits
        ]
        gates = []
        # Apply user-defined operations
        for op in operations:
            gates.append(
                kron(
                    *pad,
                    op(control, self.get_param(params, layer, rep, control)),
                    op(target, self.get_param(params, layer, rep, target)),
                )
            )
        gates.append(entangler(control, target))

        # Conditionally apply negative gates in reverse order
        if self.use_dagger:  # Only apply negative gates if use_dagger is True
            for op in reversed(operations):
                gates.append(
                    kron(
                        *pad,
                        op(control, -self.get_param(params, layer, rep, target)),
                        op(target, -self.get_param(params, layer, rep, control)),
                    )
                )
            gates.append(entangler(control, target))

        return chain(*gates)

    def create_layer(
        self,
        layer_index: int,
        reps: int,
        current_indices: list[int],
        params: dict,
        operations: list[Any],
        entangler: Any,
        n_qubits: int,
    ) -> tuple[AbstractBlock, list[int]]:
        """
        Helper function to create a single layer of the ansatz.

        Args:
            layer_index (int): The index of the current layer.
            reps (int): Number of repetitions for this layer.
            current_indices (list[int]): Indices of qubits for the current layer.
            params (dict): Dictionary to store and retrieve parameters.
            operations (list[Any]): List of quantum operations to apply in the gates.
            entangler (Any): Entangling operation, such as CNOT or CZ.
            n_qubits (int): Total number of qubits.

        Returns:
            tuple[AbstractBlock, list[int]]: A tuple containing the quantum block for
                the layer and the target indices for the next layer.
        """
        current_layer = []
        next_indices = []  # To store the targets for the next layer
        spacing = layer_index  # Define spacing based on layer index

        if layer_index in [0, 1]:  # Special behavior for first two layers
            layer_reps = []
            for d in range(reps):
                rep_kron = []
                # Define qubit pairs based on odd/even repetition
                if d % 2 == 0:  # Even d: regular behavior
                    pairs = zip(current_indices[::2], current_indices[1::2])
                else:  # Odd d: shift downward, leaving qubits 0 and 7 free
                    pairs = zip(current_indices[1:-1:2], current_indices[2:-1:2])

                # Build the gate sequence for each pair
                for control, target in pairs:
                    gate_sequence = self.create_gate_sequence(
                        params,
                        operations,
                        entangler,
                        layer_index,
                        d,
                        control,
                        target,
                        spacing=spacing,
                        n_qubits=n_qubits,
                    )
                    rep_kron.append(gate_sequence)

                # Combine gates for this repetition using `kron`
                if rep_kron:
                    layer_reps.append(kron(*rep_kron))

            # Combine all repetitions using `chain`
            if layer_reps:
                current_layer.append(chain(*layer_reps))

        else:  # Original behavior for other layers
            for d in range(reps):
                for control, target in zip(current_indices[::2], current_indices[1::2]):
                    gate_sequence = self.create_gate_sequence(
                        params,
                        operations,
                        entangler,
                        layer_index,
                        d,
                        control,
                        target,
                        spacing=spacing,
                        n_qubits=n_qubits,
                    )
                    current_layer.append(gate_sequence)

        # Update `next_indices` with the **targets** of the current layer
        next_indices = current_indices[1::2]
        return chain(*current_layer), next_indices

    def create_circuit(
        self,
        n_inputs: int,
        n_qubits: int,
        depth: list[int],
        operations: list[Any],
        entangler: Any,
    ) -> tuple[QuantumCircuit, list[int]]:
        """
        Defines a single, continuous quantum circuit with custom repeating ansatz for each depth.

        Args:
            n_inputs (int): Number of input features.
            n_qubits (int): Total number of qubits.
            depth (list[int]): List defining the depth (repetitions) of each layer.
            operations (list[Any]): List of quantum operations to apply in the gates.
            entangler (Any): Entangling operation, such as CNOT or CZ.

        Returns:
            tuple[QuantumCircuit, list[int]]: A tuple containing the quantum circuit
                and the final target indices.
        """
        # Feature map (FM)
        fm_temp = []
        for i in range(n_inputs):
            # Calculate the start and end qubits for each input feature
            start = i * (n_qubits // n_inputs)
            end = (i + 1) * (n_qubits // n_inputs) if i != n_inputs - 1 else n_qubits
            support = tuple(range(start, end))
            fm_temp.append(
                feature_map(
                    n_qubits=len(support),
                    param=f"\u03C6_{i}",
                    op=RX,
                    fm_type="Fourier",
                    support=support,
                )
            )
        fm = kron(*fm_temp)
        tag(fm, "FM")

        ansatz_layers = []  # To store each layer of the ansatz
        params: dict[str, Parameter] = {}
        all_target_indices = []  # To store target indices for each layer

        # Define layer patterns based on depth
        layer_patterns = [
            (2**layer_index, depth[layer_index]) for layer_index in range(len(depth))
        ]

        # Initialize all qubits for the first layer
        current_indices = list(range(n_qubits))

        # Build the circuit layer by layer using the helper
        for layer_index, (_, reps) in enumerate(layer_patterns):
            if reps == 0 or len(current_indices) < 2:
                break  # Skip this layer if depth is 0 or fewer than 2 qubits remain

            layer_block, next_indices = self.create_layer(
                layer_index,
                reps,
                current_indices,
                params,
                operations,
                entangler,
                n_qubits,
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

    def create_obs(
        self, n_qubits: int, random_meas: bool, last_target_qubits: list[int]
    ) -> AbstractBlock | list[AbstractBlock]:
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
            w1 = [Parameter(f"w{i}") for i in last_target_qubits]
            obs = add(Z(i) * w for i, w in zip(last_target_qubits, w1))
        else:
            obs = add(Z(i) for i in last_target_qubits)

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

    def __init__(
        self, qgcn_list: torch.nn.ModuleList, qgcn_output_size: int = 1
    ) -> None:
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