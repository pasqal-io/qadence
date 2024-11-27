from __future__ import annotations

import itertools
from typing import Any, Type, Union

from qadence.blocks import AbstractBlock, block_is_qubit_hamiltonian, chain, kron, tag
from qadence.constructors.hamiltonians import hamiltonian_factory
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY, HamEvo
from qadence.types import Interaction, Strategy

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def hea(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    strategy: Strategy = Strategy.DIGITAL,
    **strategy_args: Any,
) -> AbstractBlock:
    """
    Factory function for the Hardware Efficient Ansatz (HEA).

    Args:
        n_qubits: number of qubits in the circuit
        depth: number of layers of the HEA
        param_prefix: the base name of the variational parameters
        support: qubit indices where the HEA is applied
        strategy: Strategy for the ansatz. One of the Strategy variants.
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
            - SDAQC | Analog: Hamiltonian generator for the
            analog entangling layer. Defaults to global ZZ Hamiltonian.
            Time parameter is considered variational.

    Returns:
        The Hardware Efficient Ansatz (HEA) circuit.

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
    support: tuple[int, ...] | None = None,
    operations: list[Type[AbstractBlock]] = [RX, RY, RX],
) -> list[AbstractBlock]:
    """Creates the layers of single qubit rotations in an HEA.

    Args:
        n_qubits: The number of qubits in the HEA.
        depth: The number of layers of rotations.
        param_prefix: The prefix for the parameter names.
        support: The qubits to apply the rotations to.
        operations: The operations to apply the rotations with.

    Returns:
        A list of digital rotation layers in the HEA.
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
    Create a 2-qubit operation for the digital HEA.

    Args:
        control (int): control qubit index
        target (int): target qubit index
        param_str (str): base for naming the variational parameter if parametric block
        op (Type[DigitalEntanglers]): 2-qubit operation (CNOT, CZ, CRX, CRY, CRZ or CPHASE)

    Returns:
        The 2-qubit digital entangler for the HEA.
    """
    if op in [CNOT, CZ]:
        return op(control, target)  # type: ignore
    elif op in [CRZ, CRY, CRX, CPHASE]:
        return op(control, target, param_str)  # type: ignore
    else:
        raise ValueError("Provided entangler not accepted for digital ansatz")


def _entanglers_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    periodic: bool = False,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> list[AbstractBlock]:
    """Creates the layers of digital entangling operations in an HEA.

    Args:
        n_qubits (int): number of qubits in the block.
        depth (int): number of layers of the HEA.
        param_prefix (str): the base name of the variational parameters
        support (tuple): qubit indices where the HEA is applied.
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ. Controlld rotations
            will have variational parameters on the rotation angles.

    Returns:
        The entanglers for the digital Hardware Efficient Ansatz (HEA).
    """
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
    support: tuple[int, ...] | None = None,
    periodic: bool = False,
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    entangler: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    """
    Construct the Digital Hardware Efficient Ansatz (HEA).

    Args:
        n_qubits (int): number of qubits in the cricuit.
        depth (int): number of layers of the HEA.
        param_prefix (str): the base name of the variational parameters
        support (tuple): qubit indices where the HEA is applied.
        periodic (bool): if the qubits should be linked periodically.
            periodic=False is not supported in emu-c.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        entangler (AbstractBlock): 2-qubit entangling operation.
            Supports CNOT, CZ, CRX, CRY, CRZ. Controlld rotations
            will have variational parameters on the rotation angles.

    Returns:
        The digital Hardware Efficient Ansatz (HEA) circuit.
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
    """
    Creates the entanglers for the sDAQC.

    Args:
        depth: The number of layers of entanglers.
        param_prefix: The prefix for the parameter names.
        entangler: The entangler to use.

    Returns:
        A list of analog entanglers for sDAQC HEA.
    """
    return [HamEvo(entangler, param_prefix + f"_t_{d}") for d in range(depth)]  # type: ignore


def hea_sDAQC(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    entangler: AbstractBlock | None = None,
) -> AbstractBlock:
    """
    Construct the Hardware Efficient Ansatz (HEA) with analog entangling layers.

    It uses step-wise digital-analog computation.

    Args:
        n_qubits (int): number of qubits in the circuit.
        depth (int): number of layers of the HEA.
        param_prefix (str): the base name of the variational parameters
        support (tuple): qubit indices where the HEA is applied.
        operations (list): list of operations to cycle through in the
            digital single-qubit rotations of each layer.
        entangler (AbstractBlock): Hamiltonian generator for the
            analog entangling layer. Defaults to global ZZ Hamiltonian.
            Time parameter is considered variational.

    Returns:
        The step-wise digital-analog Hardware Efficient Ansatz (sDA HEA) circuit.
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
