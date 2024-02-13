from __future__ import annotations

from collections import Counter
from functools import reduce

import numpy as np
import torch
from torch import Tensor

from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import PrimitiveBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import Noise
from qadence.operations import H, SDagger, X, Y, Z
from qadence.parameters import evaluate
from qadence.utils import Endianness


def get_qubit_indices_for_op(pauli_term: tuple, op: PrimitiveBlock | None = None) -> list[int]:
    """Get qubit indices for the given op in the Pauli term if any."""
    blocks = getattr(pauli_term[0], "blocks", None)
    blocks = blocks if blocks is not None else [pauli_term[0]]
    indices = [
        block.qubit_support[0] for block in blocks if (op is None) or (isinstance(block, type(op)))
    ]
    return indices


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


def pauli_z_expectation(
    pauli_decomposition: list,
    samples: list[Counter],
) -> Tensor:
    """Estimate total expectation value from samples by averaging all Pauli terms."""

    estimated_values = []
    for pauli_term in pauli_decomposition:
        support = get_qubit_indices_for_op(pauli_term)
        estim_values = empirical_average(samples=samples, support=support)
        # TODO: support for parametric observables to be tested
        estimated_values.append(estim_values * evaluate(pauli_term[1]))
    res = torch.sum(torch.stack(estimated_values), axis=0)
    return res


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


def iterate_pauli_decomposition(
    circuit: QuantumCircuit,
    param_values: dict,
    pauli_decomposition: list,
    n_shots: int,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms."""

    estimated_values = []

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
