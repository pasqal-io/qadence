from __future__ import annotations

from logging import getLogger

import torch

from qadence.blocks import AbstractBlock, KronBlock
from qadence.blocks.utils import unroll_block_with_scaling
from qadence.operations import N, Z
from qadence.parameters import Parameter, evaluate
from qadence.types import GenDAQC

from .utils import _ix_map

logger = getLogger(__name__)


def _parse_generator(
    n_qubits: int,
    generator: AbstractBlock,
    zero_tol: float,
) -> torch.Tensor:
    """
    Parses the input generator to extract the `g_jk` weights.

    of the Ising model and the respective target qubits `(j, k)`.
    """

    flat_size = int(0.5 * n_qubits * (n_qubits - 1))
    g_jk_list = torch.zeros(flat_size)
    g_jk_mat = torch.zeros(n_qubits, n_qubits)

    # This parser is heavily dependent on unroll_block_with_scaling
    gen_list = unroll_block_with_scaling(generator)

    # Now we wish to check if generator is of the form:
    # `add(g_jk * kron(op(j), op(k)) for j < k)`
    # and determine if `op = Z` or `op = N`

    gen_type_Z = []
    gen_type_N = []

    for block, scale in gen_list:
        if isinstance(scale, Parameter):
            raise TypeError("DAQC transform does not support parameterized Hamiltonians.")

        # First we check if all relevant blocks (with non-negligible scaling)
        # are of type(KronBlock), since we only admit kron(Z, Z) or kron(N, N).
        if not isinstance(block, KronBlock):
            if abs(scale) < zero_tol:
                continue
            else:
                raise TypeError(
                    "DAQC transform only supports ZZ or NN interaction Hamiltonians."
                    "Error found on block: {block}."
                )

        # Next we check and keep track of the contents of each KronBlock
        for pauli in block.blocks:
            if isinstance(pauli, Z):
                gen_type_Z.append(True)
                gen_type_N.append(False)
            elif isinstance(pauli, N):
                gen_type_N.append(True)
                gen_type_Z.append(False)
            else:
                raise ValueError(
                    "DAQC transform only supports ZZ or NN interaction Hamiltonians."
                    "Error found on block: {block}."
                )

        # We save the qubit support and interaction
        # strength of each KronBlock to be used in DAQC
        j, k = block.qubit_support
        g_jk = torch.tensor(evaluate(scale), dtype=torch.get_default_dtype())

        beta = _ix_map(n_qubits, j + 1, k + 1)

        # Flat list of interaction strength
        g_jk_list[beta - 1] += g_jk

        # Symmetric matrix of interaction strength
        g_jk_mat[j, k] += g_jk
        g_jk_mat[k, j] += g_jk

    # Finally we check if all individual interaction terms were
    # either ZZ or NN to determine the generator type.
    if torch.tensor(gen_type_Z).prod() == 1 and len(gen_type_Z) > 0:
        gen_type = GenDAQC.ZZ
    elif torch.tensor(gen_type_N).prod() == 1 and len(gen_type_N) > 0:
        gen_type = GenDAQC.NN
    else:
        raise ValueError(
            "Wrong Hamiltonian structure provided. "
            "Possible mixture of Z and N terms in the Hamiltonian."
        )

    g_jk_list[g_jk_list == 0.0] = zero_tol

    return g_jk_list, g_jk_mat, gen_type


def _check_compatibility(
    g_jk_target: torch.Tensor,
    g_jk_build: torch.Tensor,
    zero_tol: float,
) -> bool:
    """
    Checks if the build Hamiltonian is missing any interactions.

    Needed for the transformation into the requested target Hamiltonian.
    """
    for g_t, g_b in zip(g_jk_target, g_jk_build):
        if abs(g_t) > zero_tol and abs(g_b) <= zero_tol:
            raise ValueError("Incompatible interactions between target and build Hamiltonians.")
    return True
