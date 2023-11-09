from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from torch import Tensor

from qadence.backends import backend_factory
from qadence.blocks import AbstractBlock, PrimitiveBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.noise import Noise
from qadence.operations import H, SDagger, X, Y, Z, chain
from qadence.parameters import evaluate
from qadence.types import BackendName, DiffMode
from qadence.utils import Endianness


def get_qubit_indices_for_op(pauli_term: tuple, op: PrimitiveBlock | None = None) -> list[int]:
    """Get qubit indices for the given op in the Pauli term if any."""
    indices = []
    blocks = getattr(pauli_term[0], "blocks", None)
    if blocks is not None:
        for block in blocks:
            if op is None:
                indices.append(block.qubit_support[0])
            if isinstance(block, type(op)):
                indices.append(block.qubit_support[0])
    else:
        block = pauli_term[0]
        if op is None:
            indices.append(block.qubit_support[0])
        if isinstance(block, type(op)):
            indices.append(block.qubit_support[0])
    return indices


def rotate(circuit: QuantumCircuit, pauli_term: tuple) -> QuantumCircuit:
    """Rotate circuit to measurement basis and return the qubit support."""
    rotations = []

    # Mypy expects concrete types. Although there definitely should be
    # a better way to pass the operation type.
    for op, gate in [(X(0), Z), (Y(0), SDagger)]:
        qubit_indices = get_qubit_indices_for_op(pauli_term, op=op)
        for index in qubit_indices:
            rotations.append(gate(index) * H(index))
    rotated_block = chain(circuit.block, *rotations)
    return QuantumCircuit(circuit.register, rotated_block)


def get_counts(samples: list, support: list) -> list:
    """Marginalise the probablity mass function to the support."""
    counts = []
    for sample in samples:
        sample_counts = []
        for k, v in sample.items():
            sample_counts.append(Counter({"".join([k[i] for i in support]): sample[k]}))
        reduced_counts = reduce(lambda x, y: x + y, sample_counts)
        counts.append(reduced_counts)
    return counts


def empirical_average(samples: list, support: list) -> Tensor:
    """Compute the empirical average."""
    counters = get_counts(samples, support)
    expectations = []
    n_shots = np.sum(list(counters[0].values()))
    parity = -1
    for counter in counters:
        counter_exps = []
        for bitstring, count in counter.items():
            counter_exps.append(count * parity ** (np.sum([int(bit) for bit in bitstring])))
        expectations.append(np.sum(counter_exps) / n_shots)
    return torch.tensor(expectations)


def iterate_pauli_decomposition(
    circuit: QuantumCircuit,
    param_values: dict,
    pauli_decomposition: list,
    n_shots: int,
    state: Tensor | None = None,
    backend_name: BackendName = BackendName.PYQTORCH,
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms."""

    estimated_values = []

    backend = backend_factory(backend=backend_name, diff_mode=DiffMode.GPSR)
    for pauli_term in pauli_decomposition:
        if pauli_term[0].is_identity:
            estimated_values.append(evaluate(pauli_term[1], as_torch=True))
        else:
            # Get the full qubit support for the Pauli term.
            # Note: duplicates must be kept here to allow for
            # observables chaining multiple operations on the same qubit
            # such as `b = chain(Z(0), Z(0))`
            support = get_qubit_indices_for_op(pauli_term)
            # Rotate the circuit according to the given observable term.
            rotated_circuit = rotate(circuit=circuit, pauli_term=pauli_term)
            # Use the low-level backend API to avoid embedding of parameters
            # already performed at the higher QuantumModel level.
            # Therefore, parameters passed here have already been embedded.
            conv_circ = backend.circuit(rotated_circuit)
            samples = backend.sample(
                circuit=conv_circ,
                param_values=param_values,
                n_shots=n_shots,
                state=state,
                noise=noise,
                endianness=endianness,
            )
            estim_values = empirical_average(samples=samples, support=support)
            # TODO: support for parametric observables to be tested
            estimated_values.append(estim_values * evaluate(pauli_term[1]))
    res = torch.sum(torch.stack(estimated_values), axis=0)
    # Allow for automatic differentiation.
    res.requires_grad = True
    return res


def compute_expectation(
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict,
    options: dict,
    state: Tensor | None = None,
    backend_name: BackendName = BackendName.PYQTORCH,
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Basic tomography protocol with rotations.

    Given a circuit and a list of observables, apply basic tomography protocol to estimate
    the expectation values.

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
    """
    if not isinstance(observables, list):
        raise TypeError(
            "Observables must be of type <class 'List[AbstractBlock]'>. Got {}.".format(
                type(observables)
            )
        )
    n_shots = options.get("n_shots")
    if n_shots is None:
        raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int'.")
    estimated_values = []
    for observable in observables:
        pauli_decomposition = unroll_block_with_scaling(observable)
        estimated_values.append(
            iterate_pauli_decomposition(
                circuit=circuit,
                param_values=param_values,
                pauli_decomposition=pauli_decomposition,
                n_shots=n_shots,
                state=state,
                backend_name=backend_name,
                noise=noise,
                endianness=endianness,
            )
        )
    return torch.transpose(torch.vstack(estimated_values), 1, 0)
