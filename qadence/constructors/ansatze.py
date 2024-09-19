from __future__ import annotations

import itertools
from typing import Any, Type, Union

from qadence.blocks import AbstractBlock, block_is_qubit_hamiltonian, chain, kron, tag
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY, HamEvo
from qadence.types import Interaction, Strategy

from .hamiltonians import hamiltonian_factory

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def hea(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] = None,
    strategy: Strategy = Strategy.DIGITAL,
    **strategy_args: Any,
) -> AbstractBlock:
    """
    Factory function for the Hardware Efficient Ansatz (HEA).

    Args:
        n_qubits: number of qubits in the block
        depth: number of layers of the HEA
        param_prefix: the base name of the variational parameters
        support: qubit indexes where the HEA is applied
        strategy: Strategy.Digital or Strategy.DigitalAnalog
        **strategy_args: see below

    Keyword Arguments:
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer. Valid for
            Digital and DigitalAnalog HEA.
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c. Valid for only
            for Digital HEA.
        entangler (AbstractBlock):
            - Digital: 2-qubit entangling operation. Supports CNOT, CZ,
            CRX, CRY, CRZ, CPHASE. Controlled rotations will have variational
            parameters on the rotation angles.
            - DigitaAnalog | Analog: Hamiltonian generator for the
            analog entangling layer. Defaults to global ZZ Hamiltonian.
            Time parameter is considered variational.

    Examples:
    ```python exec="on" source="material-block" result="json"
    from qadence import RZ, RX
    from qadence import hea

    # create the circuit
    n_qubits, depth = 2, 4
    ansatz = hea(
        n_qubits=n_qubits,
        depth=depth,
        strategy="sDAQC",
        operations=[RZ,RX,RZ]
    )
    ```
    """

    if support is None:
        support = tuple(range(n_qubits))

    hea_func_dict = {
        Strategy.DIGITAL: hea_digital,
        Strategy.SDAQC: hea_sDAQC,
        Strategy.BDAQC: hea_bDAQC,
        Strategy.ANALOG: hea_analog,
    }

    try:
        hea_func = hea_func_dict[strategy]
    except KeyError:
        raise KeyError(f"Strategy {strategy} not recognized.")

    hea_block: AbstractBlock = hea_func(
        n_qubits=n_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        **strategy_args,
    )  # type: ignore

    return hea_block


#############
## DIGITAL ##
#############


def _rotations_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] = None,
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
        raise ValueError("Provided entangler not accepted for digital HEA.")


def _entanglers_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] = None,
    periodic: bool = False,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> list[AbstractBlock]:
    """Creates the layers of digital entangling operations in an HEA."""
    if support is None:
        support = tuple(range(n_qubits))
    iterator = itertools.count()
    ent_list: list[AbstractBlock] = []
    for d in range(depth):
        ents = []
        ents.append(
            kron(
                _entangler(
                    control=support[n],
                    target=support[n + 1],
                    param_str=param_prefix + f"_ent_{next(iterator)}",
                    op=entangler,
                )
                for n in range(n_qubits)
                if not n % 2 and n < n_qubits - 1
            )
        )
        if n_qubits > 2:
            ents.append(
                kron(
                    _entangler(
                        control=support[n],
                        target=support[(n + 1) % n_qubits],
                        param_str=param_prefix + f"_ent_{next(iterator)}",
                        op=entangler,
                    )
                    for n in range(n_qubits - (not periodic))
                    if n % 2
                )
            )
        ent_list.append(chain(*ents))
    return ent_list


def hea_digital(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    periodic: bool = False,
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    support: tuple[int, ...] = None,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    """
    Construct the Digital Hardware Efficient Ansatz (HEA).

    Args:
        n_qubits (int): number of qubits in the block.
        depth (int): number of layers of the HEA.
        param_prefix (str): the base name of the variational parameters
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        support (tuple): qubit indexes where the HEA is applied.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ. Controlld rotations
            will have variational parameters on the rotation angles.
    """
    try:
        if entangler not in [CNOT, CZ, CRX, CRY, CRZ, CPHASE]:
            raise ValueError(
                "Please provide a valid two-qubit entangler operation for digital HEA."
            )
    except TypeError:
        raise ValueError("Please provide a valid two-qubit entangler operation for digital HEA.")

    rot_list = _rotations_digital(
        n_qubits=n_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        operations=operations,
    )

    ent_list = _entanglers_digital(
        n_qubits=n_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        periodic=periodic,
        entangler=entangler,
    )

    layers = []
    for d in range(depth):
        layers.append(rot_list[d])
        layers.append(ent_list[d])
    return tag(chain(*layers), "HEA")


###########
## sDAQC ##
###########


def _entanglers_analog(
    depth: int,
    param_prefix: str = "theta",
    entangler: AbstractBlock | None = None,
) -> list[AbstractBlock]:
    return [HamEvo(entangler, param_prefix + f"_t_{d}") for d in range(depth)]


def hea_sDAQC(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    support: tuple[int, ...] = None,
    entangler: AbstractBlock | None = None,
) -> AbstractBlock:
    """
    Construct the Hardware Efficient Ansatz (HEA) with analog entangling layers.

    It uses step-wise digital-analog computation.

    Args:
        n_qubits (int): number of qubits in the block.
        depth (int): number of layers of the HEA.
        param_prefix (str): the base name of the variational parameters
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        support (tuple): qubit indexes where the HEA is applied.
        entangler (AbstractBlock): Hamiltonian generator for the
            analog entangling layer. Defaults to global ZZ Hamiltonian.
            Time parameter is considered variational.
    """

    # TODO: Add qubit support
    if entangler is None:
        entangler = hamiltonian_factory(n_qubits, interaction=Interaction.NN)
    try:
        if not block_is_qubit_hamiltonian(entangler):
            raise ValueError(
                "Please provide a valid Pauli Hamiltonian generator for digital-analog HEA."
            )
    except NotImplementedError:
        raise ValueError(
            "Please provide a valid Pauli Hamiltonian generator for digital-analog HEA."
        )

    rot_list = _rotations_digital(
        n_qubits=n_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        operations=operations,
    )

    ent_list = _entanglers_analog(
        depth=depth,
        param_prefix=param_prefix,
        entangler=entangler,
    )

    layers = []
    for d in range(depth):
        layers.append(rot_list[d])
        layers.append(ent_list[d])
    return tag(chain(*layers), "HEA-sDA")


###########
## bDAQC ##
###########


def hea_bDAQC(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


############
## ANALOG ##
############


def hea_analog(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def alt(
    n_qubits: int,
    m_block_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] = None,
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

    hea_block: AbstractBlock = alt_func(
        n_qubits=n_qubits,
        m_block_qubits=m_block_qubits,
        depth=depth,
        param_prefix=param_prefix,
        support=support,
        **strategy_args,
    )  # type: ignore

    return hea_block


#################
## DIGITAL ALT ##
#################


def _entanglers_block_digital(
    n_qubits: int,
    m_block_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] = None,
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
    support: tuple[int, ...] = None,
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

    ent_list = _entanglers_block_digital(
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
