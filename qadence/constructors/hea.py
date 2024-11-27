from __future__ import annotations

from typing import Any, Type

from qadence.blocks import AbstractBlock, block_is_qubit_hamiltonian, chain, tag
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY
from qadence.types import Interaction, Strategy

from .ansatze import DigitalEntanglers, _entanglers_analog, _entanglers_digital, _rotations_digital
from .hamiltonians import hamiltonian_factory


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


def hea_digital(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    periodic: bool = False,
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    support: tuple[int, ...] | None = None,
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


def hea_sDAQC(
    n_qubits: int,
    depth: int = 1,
    param_prefix: str = "theta",
    operations: list[type[AbstractBlock]] = [RX, RY, RX],
    support: tuple[int, ...] | None = None,
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
