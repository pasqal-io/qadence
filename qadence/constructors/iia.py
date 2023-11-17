from __future__ import annotations

from typing import Any, Type, Union

import torch

from qadence.blocks import AbstractBlock, KronBlock, chain, kron, tag
from qadence.operations import CNOT, CRX, CRY, CRZ, CZ, RX, RY
from qadence.parameters import Parameter

DigitalEntanglers = Union[CNOT, CZ, CRZ, CRY, CRX]


def _entangler(
    control: int,
    target: int,
    param_str: str,
    entangler: Type[DigitalEntanglers] = CNOT,
) -> AbstractBlock:
    if entangler in [CNOT, CZ]:
        return entangler(control, target)  # type: ignore
    elif entangler in [CRZ, CRY, CRX]:
        param = Parameter(param_str, value=0.0, trainable=True)
        return entangler(control, target, param)  # type: ignore
    else:
        raise ValueError("Provided entangler not accepted for digital ansatz.")


def _rotations(
    n_qubits: int,
    layer: int,
    side: str,
    param_str: str,
    values: list[float | torch.Tensor],
    ops: list[type[AbstractBlock]] = [RX, RY],
) -> list[KronBlock]:
    if side == "left":
        idx = lambda x: x  # noqa: E731
    elif side == "right":
        idx = lambda x: len(ops) - x - 1  # noqa: E731
        ops = list(reversed(ops))
    else:
        raise ValueError("Please provide either 'left' or 'right'")

    rot_list = []
    for i, gate in enumerate(ops):
        rot_list.append(
            kron(
                gate(
                    target=n,  # type: ignore [call-arg]
                    parameter=Parameter(  # type: ignore [call-arg]
                        name=param_str + f"_{layer}{n + n_qubits * idx(i)}",
                        value=values[n + n_qubits * idx(i)],
                        trainable=True,
                    ),
                )
                for n in range(n_qubits)
            )
        )

    return rot_list


def identity_initialized_ansatz(
    n_qubits: int,
    depth: int = 1,
    rotations: Any = [RX, RY],
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

        left_rotations = _rotations(
            n_qubits=n_qubits,
            layer=layer,
            side="left",
            param_str="alpha",
            values=alpha,
            ops=rotations,
        )

        param_prefix = "theta_ent_"
        if not periodic:
            left_entanglers = [
                chain(
                    _entangler(
                        control=n,
                        target=n + 1,
                        param_str=param_prefix + f"_{layer}{n}",
                        entangler=entangler,
                    )
                    for n in range(n_qubits - 1)
                )
            ]
        else:
            left_entanglers = [
                chain(
                    _entangler(
                        control=n,
                        target=(n + 1) % n_qubits,
                        param_str=param_prefix + f"_{layer}{n}",
                        entangler=entangler,
                    )
                    for n in range(n_qubits)
                )
            ]

        centre_rotations = [
            kron(
                RX(target=n, parameter=Parameter(name="gamma" + f"_{layer}{n}", value=gamma[n]))
                for n in range(n_qubits)
            )
        ]

        right_entanglers = reversed(*left_entanglers)

        right_rotations = _rotations(
            n_qubits=n_qubits,
            layer=layer,
            side="right",
            param_str="beta",
            values=beta,
            ops=rotations,
        )

        krons = [
            *left_rotations,
            *left_entanglers,
            *centre_rotations,
            *right_entanglers,
            *right_rotations,
        ]

        initialized_layers.append(tag(chain(*krons), tag=f"BPMA-{layer}"))

    return chain(*initialized_layers)
