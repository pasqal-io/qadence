from __future__ import annotations

from torch import Tensor

from qadence.blocks import AbstractBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.circuit import QuantumCircuit
from qadence.measurements.utils import iterate_pauli_decomposition
from qadence.noise import Noise
from qadence.types import BackendName
from qadence.utils import Endianness


def compute_expectation(
    circuit: QuantumCircuit,
    observables: list[AbstractBlock],
    param_values: dict,
    options: dict,
    state: Tensor | None = None,
    backend: BackendName = BackendName.PYQTORCH,
    noise: Noise | None = None,
    endianness: Endianness = Endianness.BIG,
) -> Tensor:
    """Basic tomography protocol with rotations.

    Given a circuit and a list of observables, apply basic tomography protocol to estimate
    the expectation values.

    Args:
        observables (list[AbstractBlock]): a list of observables
            to estimate the expectation values from.
        options (dict): a dict of options for the measurement protocol.
            Here, shadow_size (int), accuracy (float) and confidence (float) are supported.
        samples (List | None): List of samples against which expectation value is to be computed
    """

    ## add test to see if all observables are in the Z basis
    if not isinstance(observables, list):
        raise TypeError(
            "Observables must be of type <class 'List[AbstractBlock]'>. Got {}.".format(
                type(observables)
            )
        )

    pauli_decomposition = unroll_block_with_scaling(observables[0])
    return iterate_pauli_decomposition(pauli_decomposition, options["samples"])
