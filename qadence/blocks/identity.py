from __future__ import annotations

import torch

from qadence.blocks import AbstractBlock, chain, kron, tag
from qadence.operations import CNOT, RX, RY
from qadence.parameters import Parameter


def identity_block(
    n_qubits: int,
    n_layers: int = 1,
    ops: list[type[AbstractBlock]] = [RX, RY],
    periodic: bool = False,
) -> AbstractBlock:
    """
    Identity block for barren plateau mitigation.
    The initial configuration of this block is equal to an identity unitary
    but can be trained in the same fashion as other ansatzes reaching same level
    of expressivity.
    """

    initialized_layers = []
    for layer in range(n_layers):
        alpha = 2 * torch.pi * torch.rand(n_qubits * len(ops))
        theta = torch.zeros(n_qubits)
        beta = -alpha
        left_rotations = [
            kron(
                gate(
                    n,  # type: ignore [arg-type]
                    Parameter("alpha" + f"_{layer}{n + n_qubits*i}", value=alpha[n + n_qubits * i]),
                )
                for n in range(n_qubits)
            )
            for i, gate in enumerate(ops)
        ]

        if not periodic:
            left_cnots = [chain(CNOT(n, n + 1) for n in range(n_qubits - 1))]
        else:
            left_cnots = [chain(CNOT(n, (n + 1) % n_qubits) for n in range(n_qubits))]

        centre_rotations = [
            kron(
                RX(n, Parameter("theta" + f"_{layer}{n}", value=theta[n])) for n in range(n_qubits)
            )
        ]

        right_cnots = reversed(*left_cnots)

        right_rotations = [
            kron(
                gate(
                    n,  # type: ignore [arg-type]
                    Parameter(
                        "beta" + f"_{layer}{n + n_qubits*(len(ops)-i-1)}",
                        value=beta[n + n_qubits * (len(ops) - i - 1)],
                    ),
                )
                for n in range(n_qubits)
            )
            for i, gate in enumerate(reversed(ops))
        ]

        krons = [
            *left_rotations,
            *left_cnots,
            *centre_rotations,
            *right_cnots,
            *right_rotations,
        ]

        initialized_layers.append(tag(chain(*krons), tag=f"BPMA-{layer}"))

    return chain(*initialized_layers)
