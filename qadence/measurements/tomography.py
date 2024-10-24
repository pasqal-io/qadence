from __future__ import annotations

import torch
from torch import Tensor

from qadence.backend import Backend
from qadence.backends.pyqtorch import Backend as PyQBackend
from qadence.blocks import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.engines.differentiable_backend import DifferentiableBackend
from qadence.measurements.utils import iterate_pauli_decomposition
from qadence.noise import NoiseHandler
from qadence.utils import Endianness


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
                backend=backend,
                noise=noise,
                endianness=endianness,
            )
        )
    return torch.transpose(torch.vstack(estimated_values), 1, 0)
