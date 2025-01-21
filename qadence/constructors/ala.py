from __future__ import annotations

import itertools
from typing import Any, Type, Union

from qadence.blocks import AbstractBlock, chain, kron, tag
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY
from qadence.types import Strategy

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def ala(
    n_qubits: int,
    m_block_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    strategy: Strategy = Strategy.DIGITAL,
    **strategy_args: Any,
) -> AbstractBlock:
    """
    Factory function for the alternating layer ansatz (ala).

    Args:
        n_qubits: number of qubits in the circuit
        m_block_qubits: number of qubits in the local entangling block
        depth: number of layers of the alternating layer ansatz
        param_prefix: the base name of the variational parameters
        support: qubit indices where the ala is applied
        strategy: Strategy for the ansatz. One of the Strategy variants.
        **strategy_args: see below

    Keyword Arguments:
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer. Valid for
            Digital .
        entangler (AbstractBlock):
            - Digital: 2-qubit entangling operation. Supports CNOT, CZ,
            CRX, CRY, CRZ, CPHASE. Controlled rotations will have variational
            parameters on the rotation angles.
            - SDAQC | BDAQC: Hamiltonian generator for the analog entangling
                layer. Must be an m-qubit operator where m is the size of the
                local entangling block. Defaults to a ZZ interaction.

    Returns:
        The Alternating Layer Ansatz (ALA) circuit.
    """

    if support is None:
        support = tuple(range(n_qubits))

    ala_func_dict = {
        Strategy.DIGITAL: ala_digital,
        Strategy.SDAQC: ala_sDAQC,
        Strategy.BDAQC: ala_bDAQC,
        Strategy.ANALOG: ala_analog,
    }

    try:
        ala_func = ala_func_dict[strategy]
    except KeyError:
        raise KeyError(f"Strategy {strategy} not recognized.")

    ala_block: AbstractBlock = ala_func(
        n_qubits=n_qubits,
        m_block_qubits=m_block_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        **strategy_args,
    )  # type: ignore

    return ala_block


#################
## DIGITAL ALA ##
#################


def _rotations_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    operations: list[Type[AbstractBlock]] = [RX, RY, RX],
) -> list[AbstractBlock]:
    """Creates the layers of single qubit rotations in an Alternating Layer Ansatz.

    Args:
        n_qubits: The number of qubits in the Alternating Layer Ansatz.
        depth: The number of layers of rotations.
        param_prefix: The prefix for the parameter names.
        support: The qubits to apply the rotations to.
        operations: The operations to apply the rotations with.

    Returns:
        A list of digital rotation layers for the Alternating Layer Ansatz.
    """
    if support is None:
        support = tuple(range(n_qubits))
    iterator = itertools.count()
    rot_list: list[AbstractBlock] = []
    for d in range(depth):
        rots = [
            kron(
                gate(support[n], param_prefix + f"_{next(iterator)}")  # type: ignore [arg-type]
                for n in range(n_qubits)
            )
            for gate in operations
        ]
        rot_list.append(chain(*rots))
    return rot_list


def _entangler(
    control: int,
    target: int,
    param_str: str,
    op: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    """
    Creates the entangler for a single qubit in an Alternating Layer Ansatz.

    Args:
        control: The control qubit.
        target: The target qubit.
        param_str: The parameter string.
        op: The entangler to use.

    Returns:
        The 2-qubit digital entangler for the Alternating Layer Ansatz.
    """
    if op in [CNOT, CZ]:
        return op(control, target)  # type: ignore
    elif op in [CRZ, CRY, CRX, CPHASE]:
        return op(control, target, param_str)  # type: ignore
    else:
        raise ValueError("Provided entangler not accepted for digital alternating layer ansatz")


def _entanglers_ala_block_digital(
    n_qubits: int,
    m_block_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> list[AbstractBlock]:
    """
    Creates the entanglers for an Alternating Layer Ansatz.

    Args:
        n_qubits: The number of qubits in the Alternating Layer Ansatz.
        m_block_qubits: The number of qubits in each block.
        depth: The number of layers of entanglers.
        param_prefix: The prefix for the parameter names.
        support (tuple): qubit indices where the HEA is applied.
        entangler: The entangler to use.

    Returns:
        The entanglers for the Alternating Layer Ansatz.
    """
    if support is None:
        support = tuple(range(n_qubits))
    iterator = itertools.count()
    ent_list: list[AbstractBlock] = []

    for d in range(depth):
        start_i = 0 if not d % 2 else -m_block_qubits // 2
        ents = [
            kron(
                _entangler(
                    control=support[i + j],
                    target=support[i + j + 1],
                    param_str=param_prefix + f"_ent_{next(iterator)}",
                    op=entangler,
                )
                for j in range(start_j, m_block_qubits, 2)
                for i in range(start_i, n_qubits, m_block_qubits)
                if i + j + 1 < n_qubits and j + 1 < m_block_qubits and i + j >= 0
            )
            for start_j in [i for i in range(2) if m_block_qubits > 2 or i == 0]
        ]

        ent_list.append(chain(*ents))
    return ent_list


def ala_digital(
    n_qubits: int,
    m_block_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    operations: list[type[AbstractBlock]] = [RX, RY],
    entangler: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    """
    Construct the digital alternating layer ansatz (ALA).

    Args:
        n_qubits (int): number of qubits in the circuit.
        m_block_qubits (int): number of qubits in the local entangling block.
        depth (int): number of layers of the ALA.
        param_prefix (str): the base name of the variational parameters
        support (tuple): qubit indices where the ALA is applied.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ. Controlld rotations
            will have variational parameters on the rotation angles.

    Returns:
        The digital Alternating Layer Ansatz (ALA) circuit.
    """

    try:
        if entangler not in [CNOT, CZ, CRX, CRY, CRZ, CPHASE]:
            raise ValueError(
                "Please provide a valid two-qubit entangler operation for digital ALA."
            )
    except TypeError:
        raise ValueError("Please provide a valid two-qubit entangler operation for digital ALA.")

    rot_list = _rotations_digital(
        n_qubits=n_qubits,
        depth=depth,
        support=support,
        param_prefix=param_prefix,
        operations=operations,
    )

    ent_list = _entanglers_ala_block_digital(
        n_qubits,
        m_block_qubits,
        param_prefix=param_prefix + "_ent",
        depth=depth,
        support=support,
        entangler=entangler,
    )

    layers = []
    for d in range(depth):
        layers.append(rot_list[d])
        layers.append(ent_list[d])

    return tag(chain(*layers), "ALA")


#################
## sdaqc ALA ##
#################
def ala_sDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


#################
## bdaqc ALA ##
#################
def ala_bDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


#################
## analog ALA ##
#################
def ala_analog(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
