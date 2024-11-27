from __future__ import annotations

import itertools
from typing import Type, Union

from qadence.blocks import AbstractBlock, chain, kron
from qadence.operations import CNOT, CPHASE, CRX, CRY, CRZ, CZ, RX, RY, HamEvo

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


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
        raise ValueError("Provided entangler not accepted for digital ansatz")


def _entanglers_digital(
    n_qubits: int,
    depth: int,
    param_prefix: str = "theta",
    support: tuple[int, ...] | None = None,
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


def _entanglers_analog(
    depth: int,
    param_prefix: str = "theta",
    entangler: AbstractBlock | None = None,
) -> list[AbstractBlock]:
    return [HamEvo(entangler, param_prefix + f"_t_{d}") for d in range(depth)]  # type: ignore
