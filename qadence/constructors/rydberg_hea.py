from __future__ import annotations

from typing import Type, Union

import sympy

import qadence as qd
from qadence.blocks import AddBlock, ChainBlock, add, chain
from qadence.constructors import hamiltonian_factory
from qadence.operations import N, X, Y, Z
from qadence.parameters import Parameter, VariationalParameter

TPauliOp = Union[Type[X], Type[Y], Type[Z], Type[N]]


def _amplitude_map(
    n_qubits: int,
    pauli_op: TPauliOp,
    weights: list[Parameter] | list[float] | None = None,
) -> AddBlock:
    """Create an generator equivalent to a laser amplitude mapping on the device.

    Basically, given a certain quantum operation `pauli_op`, this routine constructs
    the following generator:

        H = sum_i^N w_i OP(i)

    where the weights are variational parameters

    Args:
        n_qubits: number of qubits
        pauli_op: type of Pauli operation to use when creating the generator
        weights: list of variational paramters with the weights

    Returns:
        A block with the Hamiltonian generator
    """
    if weights is None:
        return add(pauli_op(j) for j in range(n_qubits))
    else:
        assert len(weights) <= n_qubits, "Wrong weights supplied"
        return add(w * pauli_op(j) for j, w in enumerate(weights))  # type:ignore [operator]


def rydberg_hea_layer(
    register: qd.Register,
    tevo_drive: Parameter | float,
    tevo_det: Parameter | float,
    tevo_wait: Parameter | float,
    phase: Parameter | float | None = None,
    detunings: list[Parameter] | list[float] | None = None,
    drives: list[Parameter] | list[float] | None = None,
    drive_scaling: float = 1.0,
) -> ChainBlock:
    """A single layer of the Rydberg hardware efficient ansatz.

    Args:
        register: the input register with atomic coordinates needed to build the interaction.
        tevo_drive: a variational parameter for the duration of the drive term of
            the Hamiltonian generator, including optional semi-local addressing
        tevo_det: a variational parameter for the duration of the detuning term of the
            Hamiltonian generator, including optional semi-local addressing
        tevo_wait: a variational parameter for the duration of the waiting
            time with interaction only
        phase: a variational parameter representing the global phase. If None, the
            global phase is set to 0 which results in a drive term in sigma^x only. Otherwise
            both sigma^x and sigma^y terms will be present
        detunings: a list of parameters with the weights of the locally addressed
            detuning terms. These are variational parameters which are tuned by the optimizer
        drives: a list of parameters with the weights of the locally addressed
            drive terms. These are variational parameters which are tuned by the optimizer
        drive_scaling: a scaling term to be added to the drive Hamiltonian generator

    Returns:
        A block with a single layer of Rydberg HEA
    """
    n_qubits = register.n_qubits

    drive_x = _amplitude_map(n_qubits, qd.X, weights=drives)
    drive_y = _amplitude_map(n_qubits, qd.Y, weights=drives)
    detuning = _amplitude_map(n_qubits, qd.N, weights=detunings)
    interaction = hamiltonian_factory(register, qd.Interaction.NN)

    # drive and interaction are not commuting thus they need to be
    # added directly into the final Hamiltonian generator
    if phase is not None:
        generator = (
            drive_scaling * sympy.cos(phase) * drive_x
            - drive_scaling * sympy.sin(phase) * drive_y
            + interaction
        )
    else:
        generator = drive_scaling * drive_x + interaction

    return chain(
        qd.HamEvo(generator, tevo_drive),
        # detuning and interaction are commuting, so they
        # can be ordered arbitrarily and treated separately
        qd.HamEvo(interaction, tevo_wait),
        qd.HamEvo(detuning, tevo_det),
    )


def rydberg_hea(
    register: qd.Register,
    n_layers: int = 1,
    addressable_detuning: bool = True,
    addressable_drive: bool = False,
    tunable_phase: bool = False,
    additional_prefix: str = None,
) -> qd.blocks.ChainBlock:
    """Hardware efficient ansatz for neutral atom (Rydberg) platforms.

    This constructor implements a variational ansatz which is very close to
    what is implementable on 2nd generation PASQAL quantum devices. In particular,
    it implements evolution over a specific Hamiltonian which can be realized on
    the device. This Hamiltonian contains:

    * an interaction term given by the standard NN interaction and determined starting
        from the positions in the input register: Hᵢₙₜ = ∑ᵢⱼ C₆/rᵢⱼ⁶ nᵢnⱼ

    * a detuning term which corresponding to a n_i = (1+sigma_i^z)/2 applied to
        all the qubits. If the `addressable_detuning` flag is set to True, the routine
        effectively a local n_i = (1+sigma_i^z)/2 term in the
        evolved Hamiltonian with a different coefficient for each atom. These
        coefficients determine a local addressing pattern for the detuning on a subset
        of the qubits. In this routine, the coefficients are variational parameters
        and they will therefore be optimized at each optimizer step

    * a drive term which corresponding to a sigma^x evolution operation applied to
        all the qubits. If the `addressable_drive` flag is set to True, the routine
        effectively a local sigma_i^x term in the evolved Hamiltonian with a different
        coefficient for each atom. These coefficients determine a local addressing pattern
        for the drive on a subset of the qubits. In this routine, the coefficients are
        variational parameters and they will therefore be optimized at each optimizer step

    * if the `tunable_phase` flag is set to True, the drive term is modified in the following
        way: drive = cos(phi) * sigma^x - sin(phi) * sigma^y
        The addressable pattern above is maintained and the phase is considered just as an
        additional variational parameter which is optimized with the rest

    Notice that, on real devices, the coefficients assigned to each qubit in both the detuning
    and drive patterns should be non-negative and they should always sum to 1. This is not the
    case for the implementation in this routine since the coefficients (weights) do not have any
    constraint. Therefore, this HEA is not completely realizable on neutral atom devices.

    Args:
        register: the input atomic register with Cartesian coordinates.
        n_layers: number layers in the HEA, each layer includes a drive, detuning and
            pure interaction pulses whose is a variational parameter
        addressable_detuning: whether to turn on the trainable semi-local addressing pattern
            on the detuning (n_i terms in the Hamiltonian)
        addressable_drive: whether to turn on the trainable semi-local addressing pattern
            on the drive (sigma_i^x terms in the Hamiltonian)
        tunable_phase: whether to have a tunable phase to get both sigma^x and sigma^y rotations
            in the drive term. If False, only a sigma^x term will be included in the drive part
            of the Hamiltonian generator
        additional_prefix: an additional prefix to attach to the parameter names

    Returns:
        The Rydberg HEA block
    """
    n_qubits = register.n_qubits
    prefix = "" if additional_prefix is None else "_" + additional_prefix

    detunings = None
    # add a detuning pattern locally addressing the atoms
    if addressable_detuning:
        detunings = [qd.VariationalParameter(f"detmap_{j}") for j in range(n_qubits)]

    drives = None
    # add a drive pattern locally addressing the atoms
    if addressable_drive:
        drives = [qd.VariationalParameter(f"drivemap_{j}") for j in range(n_qubits)]

    phase = None
    if tunable_phase:
        phase = qd.VariationalParameter("phase")

    return chain(
        rydberg_hea_layer(
            register,
            VariationalParameter(f"At{prefix}_{layer}"),
            VariationalParameter(f"Omega{prefix}_{layer}"),
            VariationalParameter(f"wait{prefix}_{layer}"),
            detunings=detunings,
            drives=drives,
            phase=phase,
        )
        for layer in range(n_layers)
    )
