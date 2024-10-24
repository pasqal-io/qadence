from __future__ import annotations

from collections import Counter
from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import torch
from sympy import Basic
from torch import Tensor

from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock, PrimitiveBlock, chain
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.noise import NoiseHandler
from qadence.operations import H, SDagger, X, Y, Z
from qadence.parameters import evaluate
from qadence.utils import Endianness


def get_qubit_indices_for_op(
    pauli_term: Tuple[AbstractBlock, Basic], op: PrimitiveBlock | None = None
) -> List[int]:
    """Get qubit indices for the given op in the Pauli term if any.

    Args:
        pauli_term: Tuple of a Pauli block and a parameter.
        op: Tuple of Primitive blocks or None.

    Returns: A list of integers representing qubit indices.
    """
    blocks = getattr(pauli_term[0], "blocks", None)
    blocks = blocks if blocks is not None else [pauli_term[0]]
    indices = [
        block.qubit_support[0] for block in blocks if (op is None) or (isinstance(block, type(op)))
    ]
    return indices


def get_counts(samples: List, support: List[int]) -> List[Counter]:
    """Marginalise the probability mass function to the support.

    Args:
        samples: List of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A List[Counter] of bit strings.
    """
    return [
        reduce(
            lambda x, y: x + y,
            [Counter({"".join([k[i] for i in support]): sample[k]}) for k, v in sample.items()],
        )
        for sample in samples
    ]


def empirical_average(samples: List, support: List[int]) -> Tensor:
    """Compute the empirical average.

    Args:
        samples: List of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A torch.Tensor of the empirical average.
    """
    PARITY = -1
    counters = get_counts(samples, support)
    n_shots = np.sum(list(counters[0].values()))
    expectations = []
    for counter in counters:
        counter_exps = []
        for bitstring, count in counter.items():
            counter_exps.append(count * PARITY ** (np.sum([int(bit) for bit in bitstring])))
        expectations.append(np.sum(counter_exps) / n_shots)
    return torch.tensor(expectations)


def pauli_z_expectation(
    pauli_decomposition: List[Tuple[AbstractBlock, Basic]],
    samples: List[Counter],
) -> Tensor:
    """Estimate total expectation value from samples by averaging all Pauli terms.

    Args:
        pauli_decomposition: A list of Pauli decomposed terms.
        samples: Samples of bit string as Counters.

    Returns: A torch.Tensor of the total expectation.
    """

    estimated_values = []
    for pauli_term in pauli_decomposition:
        support = get_qubit_indices_for_op(pauli_term)
        estim_values = empirical_average(samples=samples, support=support)
        # TODO: support for parametric observables to be tested
        estimated_values.append(estim_values * evaluate(pauli_term[1]))
    res = torch.sum(torch.stack(estimated_values), axis=0)
    return res


def rotate(circuit: QuantumCircuit, pauli_term: Tuple[AbstractBlock, Basic]) -> QuantumCircuit:
    """Rotate circuit to measurement basis and return the qubit support.

    Args:
        circuit: The circuit that is executed.
        pauli_term: Tuple of a Pauli term and a parameter.

    Returns: Rotated QuantumCircuit.
    """

    rotations = []

    for op, gate in [(X(0), Z), (Y(0), SDagger)]:
        qubit_indices = get_qubit_indices_for_op(pauli_term, op=op)
        for index in qubit_indices:
            rotations.append(gate(index) * H(index))
    rotated_block = chain(circuit.block, *rotations)
    return QuantumCircuit(circuit.register, rotated_block)


def iterate_pauli_decomposition(
    circuit: QuantumCircuit,
    param_values: Dict[str, Tensor],
    pauli_decomposition: List[Tuple[AbstractBlock, Basic]],
    n_shots: int,
    state: Tensor | None = None,
    backend: Backend | DifferentiableBackend = PyQBackend(),
    noise: NoiseHandler | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms.

    Args:
        circuit: The circuit that is executed.
        param_values: Parameters of the circuit.
        pauli_decomposition: A list of Pauli decomposed terms.
        n_shots: Number of shots to sample.
        state: Initial state.
        backend: A backend for circuit execution.
        noise: A noise model to use.
        endianness: Endianness of the resulting bit strings.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

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
