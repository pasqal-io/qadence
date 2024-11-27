from __future__ import annotations

import itertools
from typing import Any, Type, Union

from qadence.blocks import AbstractBlock, chain, kron, tag
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY
from qadence.types import Strategy

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def alt(
    n_qubits: int,
    m_block_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    strategy: Strategy = Strategy.DIGITAL,
    **strategy_args: Any,
) -> AbstractBlock:
    """
    Factory function for the alternating layer ansatz (alt).

    Args:
        n_qubits: number of qubits in the block
        m_block_qubits: number of qubits in the local entangling block
        depth: number of layers of the alt
        param_prefix: the base name of the variational parameters
        support: qubit indexes where the alt is applied
        strategy: Strategy.Digital or Strategy.DigitalAnalog
        **strategy_args: see below

    Keyword Arguments:
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer. Valid for
            Digital .
        entangler (AbstractBlock):
            - Digital: 2-qubit entangling operation. Supports CNOT, CZ,
            CRX, CRY, CRZ, CPHASE. Controlled rotations will have variational
            parameters on the rotation angles.
    """

    if support is None:
        support = tuple(range(n_qubits))

    alt_func_dict = {
        Strategy.DIGITAL: alt_digital,
        Strategy.SDAQC: alt_sDAQC,
        Strategy.BDAQC: alt_bDAQC,
        Strategy.ANALOG: alt_analog,
    }

    try:
        alt_func = alt_func_dict[strategy]
    except KeyError:
        raise KeyError(f"Strategy {strategy} not recognized.")

    alt_block: AbstractBlock = alt_func(
        n_qubits=n_qubits,
        m_block_qubits=m_block_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        **strategy_args,
    )  # type: ignore

    return alt_block


#################
## DIGITAL ALT ##
#################


def _rotations_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    operations: list[Type[AbstractBlock]] = [RX, RY, RX],
) -> list[AbstractBlock]:
    """Creates the layers of single qubit rotations in an HEA."""
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
    if op in [CNOT, CZ]:
        return op(control, target)  # type: ignore
    elif op in [CRZ, CRY, CRX, CPHASE]:
        return op(control, target, param_str)  # type: ignore
    else:
        raise ValueError("Provided entangler not accepted for digital alternating block ansatz")


def _entanglers_alt_block_digital(
    n_qubits: int,
    m_block_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> list[AbstractBlock]:
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


def alt_digital(
    n_qubits: int,
    m_block_qubits: int,
    depth: int = 1,
    support: tuple[int, ...] | None = None,
    param_prefix: str = "theta",
    operations: list[type[AbstractBlock]] = [RX, RY],
    entangler: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    """
    Construct the digital alternating layer ansatz (ALT).

    Args:
        n_qubits (int): number of qubits in the ansatz.
        m_block_qubits (int): number of qubits in the local entangling block.
        depth (int): number of layers of the ALT.
        param_prefix (str): the base name of the variational parameters
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        support (tuple): qubit indexes where the ALT is applied.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ. Controlld rotations
            will have variational parameters on the rotation angles.
    """

    try:
        if entangler not in [CNOT, CZ, CRX, CRY, CRZ, CPHASE]:
            raise ValueError(
                "Please provide a valid two-qubit entangler operation for digital ALT."
            )
    except TypeError:
        raise ValueError("Please provide a valid two-qubit entangler operation for digital ALT.")

    rot_list = _rotations_digital(
        n_qubits=n_qubits,
        depth=depth,
        support=support,
        param_prefix=param_prefix,
        operations=operations,
    )

    ent_list = _entanglers_alt_block_digital(
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

    return tag(chain(*layers), "ALT")


#################
## sdaqc ALT ##
#################
def alt_sDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


#################
## bdaqc ALT ##
#################
def alt_bDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


#################
## analog ALT ##
#################
def alt_analog(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
