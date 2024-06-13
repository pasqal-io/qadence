from __future__ import annotations

from logging import getLogger

import torch

from qadence.blocks import AbstractBlock, add, chain, kron
from qadence.blocks.utils import block_is_qubit_hamiltonian
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.operations import HamEvo, I, N, X
from qadence.types import GenDAQC, Interaction, Strategy

from .gen_parser import _check_compatibility, _parse_generator
from .utils import _build_matrix_M, _ix_map

logger = getLogger(__name__)


def daqc_transform(
    n_qubits: int,
    gen_target: AbstractBlock,
    t_f: float,
    gen_build: AbstractBlock | None = None,
    zero_tol: float = 1e-08,
    strategy: Strategy = Strategy.SDAQC,
    ignore_global_phases: bool = False,
) -> AbstractBlock:
    """
    Implements the DAQC transform for representing an arbitrary 2-body Hamiltonian.

    The result is another fixed 2-body Hamiltonian.

    Reference for universality of 2-body Hamiltonians:

    -- https://arxiv.org/abs/quant-ph/0106064

    Based on the transformation for Ising (ZZ) interactions, as described in the paper

    -- https://arxiv.org/abs/1812.03637

    The transform translates a target weighted generator of the type:

        `gen_target = add(g_jk * kron(op(j), op(k)) for j < k)`

    To a circuit using analog evolutions with a fixed building block generator:

        `gen_build = add(f_jk * kron(op(j), op(k)) for j < k)`

    where `op = Z` or `op = N`.

    Args:
        n_qubits: total number of qubits to use.
        gen_target: target generator built with the structure above. The type
            of the generator will be automatically evaluated when parsing.
        t_f (float): total time for the gen_target evolution.
        gen_build: fixed generator to act as a building block. Defaults to
            constant NN: add(1.0 * kron(N(j), N(k)) for j < k). The type
            of the generator will be automatically evaluated when parsing.
        zero_tol: default "zero" for a missing interaction. Included for
            numerical reasons, see notes below.
        strategy: sDAQC or bDAQC, following definitions in the reference paper.
        ignore_global_phases: if `True` the transform does not correct the global
            phases coming from the mapping between ZZ and NN interactions.

    Notes:

        The paper follows an index convention of running from 1 to N. A few functions
        here also use that convention to be consistent with the paper. However, for qadence
        related things the indices are converted to [0, N-1].

        The case for `n_qubits = 4` is an edge case where the sign matrix is not invertible.
        There is a workaround for this described in the paper, but it is currently not implemented.

        The current implementation may result in evolution times that are both positive or
        negative. In practice, both can be represented by simply changing the signs of the
        interactions. However, for a real implementation where the interactions should remain
        fixed, the paper discusses a workaround that is not currently implemented.

        The transformation works by representing each interaction in the target hamiltonian by
        a set of evolutions using the build hamiltonian. As a consequence, some care must be
        taken when choosing the build hamiltonian. Some cases:

        - The target hamiltonian can have any interaction, as long as it is sufficiently
        represented in the build hamiltonian. E.g., if the interaction `g_01 * kron(Z(0), Z(1))`
        is in the target hamiltonian, the corresponding interaction `f_01 * kron(Z(0), Z(1))`
        needs to be in the build hamiltonian. This is checked when the generators are parsed.

        - The build hamiltonian can have any interaction, irrespectively of it being needed
        for the target hamiltonian. This is especially useful for designing local operations
        through the repeated evolution of a "global" hamiltonian.

        - The parameter `zero_tol` controls what it means for an interaction to be "missing".
        Any interaction strength smaller than `zero_tol` in the build hamiltonian will not be
        considered, and thus that interaction is missing.

        - The various ratios `g_jk / f_jk` will influence the time parameter for the various
        evolution slices, meaning that if there is a big discrepancy in the interaction strength
        for a given qubit pair (j, k), the output circuit may require the usage of hamiltonian
        evolutions with very large times.

        - A warning will be issued for evolution times larger than `1/sqrt(zero_tol)`. Evolution
        times smaller than `zero_tol` will not be represented.

    Examples:
        ```python exec="on" source="material-block" result="json"
        from qadence import Z, N, daqc_transform

        n_qubits = 3

        gen_build = 0.5 * (N(0)@N(1)) + 0.7 * (N(1)@N(2)) + 0.2 * (N(0)@N(2))

        gen_target = 0.1 * (Z(1)@Z(2))

        t_f = 2.0

        transformed_circuit = daqc_transform(
            n_qubits = n_qubits,
            gen_target = gen_target,
            t_f = t_f,
            gen_build = gen_build,
        )
        ```
    """

    ##################
    # Input controls #
    ##################

    if strategy != Strategy.SDAQC:
        raise NotImplementedError("Currently only the sDAQC transform is implemented.")

    if n_qubits == 4:
        raise NotImplementedError("DAQC transform 4-qubit edge case not implemented.")

    if gen_build is None:
        gen_build = hamiltonian_factory(n_qubits, interaction=Interaction.NN)

    try:
        if (not block_is_qubit_hamiltonian(gen_target)) or (
            not block_is_qubit_hamiltonian(gen_build)
        ):
            raise ValueError(
                "Generator block is not a qubit Hamiltonian. Only ZZ or NN interactions allowed."
            )
    except NotImplementedError:
        # Happens when block_is_qubit_hamiltonian is called on something that is not a block.
        raise TypeError(
            "Generator block is not a qubit Hamiltonian. Only ZZ or NN interactions allowed."
        )

    #####################
    # Generator parsing #
    #####################

    g_jk_target, mat_jk_target, target_type = _parse_generator(n_qubits, gen_target, 0.0)
    g_jk_build, mat_jk_build, build_type = _parse_generator(n_qubits, gen_build, zero_tol)

    # Get the global phase hamiltonian and single-qubit detuning hamiltonian
    if build_type == GenDAQC.NN:
        h_phase_build, h_sq_build = _nn_phase_and_detunings(n_qubits, mat_jk_build)

    if target_type == GenDAQC.NN:
        h_phase_target, h_sq_target = _nn_phase_and_detunings(n_qubits, mat_jk_target)

    # Time re-scalings
    if build_type == GenDAQC.ZZ and target_type == GenDAQC.NN:
        t_star = t_f / 4.0
    elif build_type == GenDAQC.NN and target_type == GenDAQC.ZZ:
        t_star = 4.0 * t_f
    else:
        t_star = t_f

    # Check if target Hamiltonian can be mapped with the build Hamiltonian
    assert _check_compatibility(g_jk_target, g_jk_build, zero_tol)

    ##################
    # DAQC Transform #
    ##################

    # Section III A of https://arxiv.org/abs/1812.03637:

    # Matrix M for the linear system, exemplified in Table I:
    matrix_M = _build_matrix_M(n_qubits)

    # Linear system mapping interaction ratios -> evolution times.
    t_slices = torch.linalg.solve(matrix_M, g_jk_target / g_jk_build) * t_star

    # ZZ-DAQC with ZZ or NN build Hamiltonian
    daqc_slices = []
    for m in range(2, n_qubits + 1):
        for n in range(1, m):
            alpha = _ix_map(n_qubits, n, m)
            t = t_slices[alpha - 1]
            if abs(t) > zero_tol:
                if abs(t) > (1 / (zero_tol**0.5)):
                    logger.warning(
                        """
Transformed circuit with very long evolution time.
Make sure your target interactions are sufficiently
represented in the build Hamiltonian."""
                    )
                x_gates = kron(X(n - 1), X(m - 1))
                analog_evo = HamEvo(gen_build, t)
                # TODO: Fix repeated X-gates
                if build_type == GenDAQC.NN:
                    # Local detuning at each DAQC layer for NN build Hamiltonian
                    sq_detuning_build = HamEvo(h_sq_build, t)
                    daqc_slices.append(chain(x_gates, sq_detuning_build, analog_evo, x_gates))
                elif build_type == GenDAQC.ZZ:
                    daqc_slices.append(chain(x_gates, analog_evo, x_gates))

    daqc_circuit = chain(*daqc_slices)

    ########################
    # Phases and Detunings #
    ########################

    if target_type == GenDAQC.NN:
        # Local detuning given a NN target Hamiltonian
        sq_detuning_target = HamEvo(h_sq_target, t_f).dagger()
        daqc_circuit = chain(sq_detuning_target, daqc_circuit)

    if not ignore_global_phases:
        if build_type == GenDAQC.NN:
            # Constant global phase given a NN build Hamiltonian
            global_phase_build = HamEvo(h_phase_build, t_slices.sum())
            daqc_circuit = chain(global_phase_build, daqc_circuit)

        if target_type == GenDAQC.NN:
            # Constant global phase and given a NN target Hamiltonian
            global_phase_target = HamEvo(h_phase_target, t_f).dagger()
            daqc_circuit = chain(global_phase_target, daqc_circuit)

    return daqc_circuit


def _nn_phase_and_detunings(
    n_qubits: int,
    mat_jk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Constant global shift, leads to a global phase
    global_shift = mat_jk.sum() / 8

    # Strength of the local detunings
    g_sq = mat_jk.sum(0) / 2

    h_phase = global_shift * kron(I(i) for i in range(n_qubits))
    h_sq = add(-1.0 * g_sq[i] * N(i) for i in range(n_qubits))

    return h_phase, h_sq
