from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from torch import Tensor

from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, chain, kron
from qadence.blocks.block_to_tensor import HMAT, IMAT, SDAGMAT, ZMAT, block_to_tensor
from qadence.blocks.composite import CompositeBlock
from qadence.blocks.primitive import PrimitiveBlock
from qadence.blocks.utils import get_pauli_blocks, unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler
from qadence.operations import X, Y, Z
from qadence.types import Endianness
from qadence.utils import P0_MATRIX, P1_MATRIX

pauli_gates = [X, Y, Z]


UNITARY_TENSOR = [
    ZMAT @ HMAT,
    SDAGMAT.squeeze(dim=0) @ HMAT,
    IMAT,
]


def identity(n_qubits: int) -> Tensor:
    return torch.eye(2**n_qubits, dtype=torch.complex128)


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


def local_shadow(sample: Counter, unitary_ids: list) -> Tensor:
    """
    Compute local shadow by inverting the quantum channel for each projector state.

    See https://arxiv.org/pdf/2002.08953.pdf
    Supplementary Material 1 and Eqs. (S17,S44).

    Expects a sample bitstring in ILO.
    """
    bitstring = list(sample.keys())[0]
    local_density_matrices = []
    for bit, unitary_id in zip(bitstring, unitary_ids):
        proj_mat = P0_MATRIX if bit == "0" else P1_MATRIX
        unitary_tensor = UNITARY_TENSOR[unitary_id].squeeze(dim=0)
        local_density_matrices.append(
            3 * (unitary_tensor.adjoint() @ proj_mat @ unitary_tensor) - identity(1)
        )
    if len(local_density_matrices) == 1:
        return local_density_matrices[0]
    else:
        return reduce(torch.kron, local_density_matrices)


def classical_shadow(
    shadow_size: int,
    circuit: QuantumCircuit,
    param_values: dict,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> list:
    shadow: list = []
    # TODO: Parallelize embarrassingly parallel loop.
    for _ in range(shadow_size):
        unitary_ids = np.random.randint(0, 3, size=(1, circuit.n_qubits))[0]
        random_unitary = [
            pauli_gates[unitary_ids[qubit]](qubit) for qubit in range(circuit.n_qubits)
        ]
        if len(random_unitary) == 1:
            random_unitary_block = random_unitary[0]
        else:
            random_unitary_block = kron(*random_unitary)
        rotated_circuit = QuantumCircuit(
            circuit.n_qubits,
            chain(circuit.block, random_unitary_block),
        )
        # Reverse endianness to get sample bitstrings in ILO.
        conv_circ = backend.circuit(rotated_circuit)
        samples = backend.sample(
            circuit=conv_circ,
            param_values=param_values,
            n_shots=1,
            state=state,
            noise=noise,
            endianness=endianness,
        )
        batched_shadow = []
        for batch in samples:
            batched_shadow.append(local_shadow(sample=batch, unitary_ids=unitary_ids))
        shadow.append(batched_shadow)

    # Reshape the shadow by batches of samples instead of samples of batches.
    # FIXME: Improve performance.
    return [list(s) for s in zip(*shadow)]


def reconstruct_state(shadow: list) -> Tensor:
    """Reconstruct the state density matrix for the given shadow."""
    return reduce(torch.add, shadow) / len(shadow)


def compute_traces(
    qubit_support: tuple,
    N: int,
    K: int,
    shadow: list,
    observable: AbstractBlock,
    endianness: Endianness = Endianness.BIG,
) -> list:
    floor = int(np.floor(N / K))
    traces = []
    # TODO: Parallelize embarrassingly parallel loop.
    for k in range(K):
        reconstructed_state = reconstruct_state(shadow=shadow[k * floor : (k + 1) * floor])
        # Reshape the observable matrix to fit the density matrix dimensions
        # by filling indentites.
        # Please note the endianness is also flipped to get results in LE.
        # FIXME: Changed below from Little to Big, double-check when Roland is back
        # FIXME: Correct these comments.
        trace = (
            (
                block_to_tensor(
                    block=observable,
                    qubit_support=qubit_support,
                    endianness=Endianness.BIG,
                ).squeeze(dim=0)
                @ reconstructed_state
            )
            .trace()
            .real
        )
        traces.append(trace)
    return traces


def estimators(
    qubit_support: tuple,
    N: int,
    K: int,
    shadow: list,
    observable: AbstractBlock,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """
    Return estimators (traces of observable times mean density matrix).

    for K equally-sized shadow partitions.

    See https://arxiv.org/pdf/2002.08953.pdf
    Algorithm 1.
    """
    # If there is no Pauli-Z operator in the observable,
    # the sample can't "hit" that measurement.
    if isinstance(observable, PrimitiveBlock):
        if type(observable) == Z:
            traces = compute_traces(
                qubit_support=qubit_support,
                N=N,
                K=K,
                shadow=shadow,
                observable=observable,
                endianness=endianness,
            )
        else:
            traces = [torch.tensor(0.0)]
    elif isinstance(observable, CompositeBlock):
        if Z in observable:
            traces = compute_traces(
                qubit_support=qubit_support,
                N=N,
                K=K,
                shadow=shadow,
                observable=observable,
                endianness=endianness,
            )
        else:
            traces = [torch.tensor(0.0)]
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
    shadow = classical_shadow(
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
        for batch in shadow:
            pauli_term_estimations = []
            for pauli_term in pauli_decomposition:
                # Get the estimators for the current Pauli term.
                # This is a tensor<float> of size K.
                estimation = estimators(
                    qubit_support=circuit.block.qubit_support,
                    N=N,
                    K=K,
                    shadow=batch,
                    observable=pauli_term[0],
                    endianness=endianness,
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
