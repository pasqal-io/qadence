from __future__ import annotations

from typing import Any, Type, Union

import torch

from qadence.blocks import AbstractBlock, chain, kron, tag
from qadence.operations import CNOT, CRX, CRY, CRZ, CZ, RX, RY
from qadence.parameters import VariationalParameter

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def _entangler(
    control: int,
    target: int,
    param_str: str,
    op: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    if op in [CNOT, CZ]:
        return op(control, target)  # type: ignore
    elif op in [CRZ, CRY, CRX]:
        param = VariationalParameter(param_str, value=0.0)
        return op(control, target, param)  # type: ignore
    else:
        raise ValueError("Provided entangler not accepted for digital ansatz.")


def identity_initialized_ansatz(
    n_qubits: int,
    depth: int = 1,
    rotations: list[type[AbstractBlock]] = [RX, RY],
    entangler: Any = CNOT,
    periodic: bool = False,
) -> AbstractBlock:
    """
    Identity block for barren plateau mitigation.

    The initial configuration of this block is equal to an identity unitary
    but can be trained in the same fashion as other ansatzes, reaching same level
    of expressivity.

    Args:
        n_qubits: number of qubits in the block
        depth: number of layers of the HEA
        rotations (list of AbstractBlocks):
            single-qubit rotations with trainable parameters
        entangler (AbstractBlock):
            2-qubit entangling operation. Supports CNOT, CZ, CRX, CRY, CRZ, CPHASE.
            Controlled rotations will have variational parameters on the rotation angles.
        periodic (bool): if the qubits should be linked periodically.
    """
    initialized_layers = []
    for layer in range(depth):
        alpha = 2 * torch.pi * torch.rand(n_qubits * len(rotations))
        gamma = torch.zeros(n_qubits)
        beta = -alpha
        left_rotations = [
            kron(
                gate(
                    n,  # type: ignore [arg-type]
                    VariationalParameter(
                        "alpha" + f"_{layer}{n + n_qubits*i}", value=alpha[n + n_qubits * i]
                    ),
                )
                for n in range(n_qubits)
            )
            for i, gate in enumerate(rotations)
        ]

        param_prefix = "theta_ent_"
        if not periodic:
            left_entanglers = [
                chain(
                    _entangler(n, n + 1, param_prefix + f"_{layer}{n}", entangler)
                    for n in range(n_qubits - 1)
                )
            ]
        else:
            left_entanglers = [
                chain(
                    _entangler(n, (n + 1) % n_qubits, param_prefix + f"_{layer}{n}", entangler)
                    for n in range(n_qubits)
                )
            ]

        centre_rotations = [
            kron(
                RX(n, VariationalParameter("gamma" + f"_{layer}{n}", value=gamma[n]))
                for n in range(n_qubits)
            )
        ]

        right_entanglers = reversed(*left_entanglers)

        right_rotations = [
            kron(
                gate(
                    n,  # type: ignore [arg-type]
                    VariationalParameter(
                        "beta" + f"_{layer}{n + n_qubits*(len(rotations)-i-1)}",
                        value=beta[n + n_qubits * (len(rotations) - i - 1)],
                    ),
                )
                for n in range(n_qubits)
            )
            for i, gate in enumerate(reversed(rotations))
        ]

        krons = [
            *left_rotations,
            *left_entanglers,
            *centre_rotations,
            *right_entanglers,
            *right_rotations,
        ]

        initialized_layers.append(tag(chain(*krons), tag=f"BPMA-{layer}"))

    return chain(*initialized_layers)
