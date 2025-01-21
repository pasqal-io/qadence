from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, KronBlock, kron
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.measurements.utils import get_qubit_indices_for_op
from qadence.noise import NoiseHandler
from qadence.operations import H, I, SDagger, X, Y, Z
from qadence.types import BackendName, Endianness

pauli_gates = [X, Y, Z]
pauli_rotations = [
    lambda index: H(index),
    lambda index: SDagger(index) * H(index),
    lambda index: None,
]

UNITARY_TENSOR = [
    HMAT,
    HMAT @ SDAGMAT,
    IMAT,
]


def _max_observable_weight(observable: AbstractBlock) -> int:
    """
    Get the maximal weight for the given observable.

    The weight is a measure of the locality of the observable,
    a count of the number of qubits on which the observable acts
    non-trivially.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eq. (S17).
    """
    pauli_decomposition = unroll_block_with_scaling(observable)
    weights = []
    for pauli_term in pauli_decomposition:
        weight = 0
        block = pauli_term[0]
        if isinstance(block, PrimitiveBlock):
            if isinstance(block, (X, Y, Z)):
                weight += 1
            weights.append(weight)
        else:
            pauli_blocks = get_pauli_blocks(block=block)
            weight = 0
            for block in pauli_blocks:
                if isinstance(block, (X, Y, Z)):
                    weight += 1
            weights.append(weight)
    return max(weights)


def maximal_weight(observables: list[AbstractBlock]) -> int:
    """Return the maximal weight if a list of observables is provided."""
    return max([_max_observable_weight(observable=observable) for observable in observables])


def number_of_samples(
    observables: list[AbstractBlock], accuracy: float, confidence: float
) -> tuple[int, ...]:
    """
    Estimate an optimal shot budget and a shadow partition size.

    to guarantee given accuracy on all observables expectation values
    within 1 - confidence range.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S23)-(S24).
    """
    max_k = maximal_weight(observables=observables)
    N = round(3**max_k * 34.0 / accuracy**2)
    K = round(2.0 * np.log(2.0 * len(observables) / confidence))
    return N, K


def nested_operator_indexing(
    idx_array: np.ndarray,
) -> list:
    """Obtain the list of rotation operators from indices.

    Args:
        idx_array (np.ndarray): Indices for obtaining the operators.

    Returns:
        list: Map of rotations.
    """
    if idx_array.ndim == 1:
        return [pauli_rotations[int(ind_pauli)](i) for i, ind_pauli in enumerate(idx_array)]  # type: ignore[abstract]
    return [nested_operator_indexing(sub_array) for sub_array in idx_array]


def kron_if_non_empty(list_operations: list) -> KronBlock | None:
    filtered_op: list = list(filter(None, list_operations))
    return kron(*filtered_op) if len(filtered_op) > 0 else None


def extract_operators(unitary_ids: np.ndarray, n_qubits: int) -> list:
    """Sample `shadow_size` rotations of `n_qubits`.

    Args:
        unitary_ids (np.ndarray): Indices for obtaining the operators.
        n_qubits (int): Number of qubits
    Returns:
        list: Pauli strings.
    """
    operations = nested_operator_indexing(unitary_ids)
    if n_qubits > 1:
        operations = [kron_if_non_empty(ops) for ops in operations]
    return operations


def classical_shadow(
    shadow_size: int,
    circuit: QuantumCircuit,
    param_values: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> tuple[np.ndarray, list[Tensor]]:
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, circuit.n_qubits))
    shadow: list = list()
    all_rotations = extract_operators(unitary_ids, circuit.n_qubits)

    initial_state = state
    backend_name = backend.name if hasattr(backend, "name") else backend.backend.name
    if backend_name == BackendName.PYQTORCH:
        # run the initial circuit without rotations
        # to save computation time
        conv_circ = backend.circuit(circuit)
        initial_state = backend.run(
            circuit=conv_circ,
            param_values=param_values,
            state=state,
            endianness=endianness,
        )
        all_rotations = [
            QuantumCircuit(circuit.n_qubits, rots) if rots else QuantumCircuit(circuit.n_qubits)
            for rots in all_rotations
        ]
    else:
        all_rotations = [
            (
                QuantumCircuit(circuit.n_qubits, circuit.block, rots)
                if rots
                else QuantumCircuit(circuit.n_qubits, circuit.block)
            )
            for rots in all_rotations
        ]

    for i in range(shadow_size):
        # Reverse endianness to get sample bitstrings in ILO.
        conv_circ = backend.circuit(all_rotations[i])
        batch_samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=1,
            state=initial_state,
            noise=noise,
            endianness=endianness,
        )
        shadow.append(batch_samples)
    bitstrings = list()
    batchsize = len(batch_samples)
    for b in range(batchsize):
        bitstrings.append([list(batch[b].keys())[0] for batch in shadow])
    bitstrings_torch = [
        1 - 2 * torch.stack([torch.tensor([int(b_i) for b_i in sample]) for sample in batch])
        for batch in bitstrings
    ]
    return unitary_ids, bitstrings_torch


def estimators(
    N: int,
    K: int,
    unitary_shadow_ids: np.ndarray,
    shadow_samples: Tensor,
    observable: AbstractBlock,
) -> Tensor:
    """
    Return trace estimators from the samples for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.
    """

    obs_qubit_support = observable.qubit_support
    if isinstance(observable, PrimitiveBlock):
        if isinstance(observable, I):
            return torch.tensor(1.0, dtype=torch.get_default_dtype())
        obs_to_pauli_index = [pauli_gates.index(type(observable))]

    elif isinstance(observable, CompositeBlock):
        obs_to_pauli_index = [
            pauli_gates.index(type(p)) for p in observable.blocks if not isinstance(p, I)  # type: ignore[arg-type]
        ]
        ind_I = set(get_qubit_indices_for_op((observable, 1.0), I(0)))
        obs_qubit_support = tuple([ind for ind in observable.qubit_support if ind not in ind_I])

    floor = int(np.floor(N / K))
    traces = []
    for k in range(K):
        indices_match = np.all(
            unitary_shadow_ids[k * floor : (k + 1) * floor, obs_qubit_support]
            == obs_to_pauli_index,
            axis=1,
        )
        if indices_match.sum() > 0:
            trace = torch.prod(
                shadow_samples[k * floor : (k + 1) * floor][indices_match][:, obs_qubit_support],
                axis=-1,
            ).sum() / sum(indices_match)
            traces.append(trace)
        else:
            traces.append(torch.tensor(0.0))
    return torch.tensor(traces, dtype=torch.get_default_dtype())


def estimations(
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict,
    shadow_size: int | None = None,
    accuracy: float = 0.1,
    confidence: float = 0.1,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Compute expectation values for all local observables using median of means."""
    # N is the estimated shot budget for the classical shadow to
    # achieve desired accuracy for all L = len(observables) within 1 - confidence probablity.
    # K is the size of the shadow partition.
    N, K = number_of_samples(observables=observables, accuracy=accuracy, confidence=confidence)
    if shadow_size is not None:
        N = shadow_size
    unitaries_ids, batch_shadow_samples = classical_shadow(
        shadow_size=N,
        circuit=circuit,
        param_values=param_values,
        state=state,
        backend=backend,
        noise=noise,
        endianness=endianness,
    )
    estimations = []
    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        batch_estimations = []
        for batch in batch_shadow_samples:
            pauli_term_estimations = []
            for pauli_term in pauli_decomposition:
                # Get the estimators for the current Pauli term.
                # This is a tensor<float> of size K.
                estimation = estimators(
                    N=N,
                    K=K,
                    unitary_shadow_ids=unitaries_ids,
                    shadow_samples=batch,
                    observable=pauli_term[0],
                )
                # Compute the median of means for the current Pauli term.
                # Weigh the median by the Pauli term scaling.
                pauli_term_estimations.append(torch.median(estimation) * pauli_term[1])
            # Sum the expectations for each Pauli term to get the expectation for the
            # current batch.
            batch_estimations.append(sum(pauli_term_estimations))
        estimations.append(batch_estimations)
    return torch.transpose(torch.tensor(estimations, dtype=torch.get_default_dtype()), 1, 0)


def compute_expectation(
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict,
    options: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """
    Construct a classical shadow of a state to estimate observable expectation values.

    Args:
        circuit (QuantumCircuit): a circuit to prepare the state.
        observables (list[AbstractBlock]): a list of observables
            to estimate the expectation values from.
        param_values (dict): a dict of values to substitute the
            symbolic parameters for.
        options (dict): a dict of options for the measurement protocol.
            Here, shadow_size (int), accuracy (float) and confidence (float) are supported.
        state (Tensor | None): an initial input state.
        backend_name (BackendName): a backend name to retrieve computations from.
        noise: A noise model to use.
        endianness: Endianness of the observable estimate.

    Returns:
        expectations (Tensor): an estimation of the expectation values.
    """
    if not isinstance(observables, list):
        raise TypeError(
            "Observables must be of type <class 'List[AbstractBlock]'>. Got {}.".format(
                type(observables)
            )
        )
    shadow_size = options.get("shadow_size", None)
    accuracy = options.get("accuracy", None)
    if shadow_size is None and accuracy is None:
        KeyError(
            "Shadow protocol requires either an option"
            "'shadow_size' of type 'int' or 'accuracy' of type 'float'."
        )
    confidence = options.get("confidence", None)
    if confidence is None:
        KeyError("Shadow protocol requires a 'confidence' kwarg of type 'float'.")
    return estimations(
        circuit=circuit,
        observables=observables,
        param_values=param_values,
        shadow_size=shadow_size,
        accuracy=accuracy,
        confidence=confidence,
        state=state,
        backend=backend,
        noise=noise,
        endianness=endianness,
    )
