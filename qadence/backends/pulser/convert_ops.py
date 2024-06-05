from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import Module

from qadence.blocks import (
    AbstractBlock,
)
from qadence.blocks.block_to_tensor import (
    block_to_tensor,
)
from qadence.utils import Endianness

from .config import Configuration


def convert_observable(
    block: AbstractBlock, n_qubits: int | None, config: Configuration = None
) -> Sequence[Module]:
    return [PulserObservable(block, n_qubits)]


class PulserObservable(Module):
    def __init__(self, block: AbstractBlock, n_qubits: int | None):
        super().__init__()
        self.block = block
        self.n_qubits = n_qubits

        if not self.block.is_parametric:
            block_mat = block_to_tensor(self.block, {}, qubit_support=block.qubit_support).squeeze(
                0
            )
            self.register_buffer("block_mat", block_mat)

    def forward(
        self,
        state: torch.Tensor,
        values: dict[str, torch.Tensor] | list = {},
        qubit_support: tuple | None = None,
        endianness: Endianness = Endianness.BIG,
    ) -> torch.Tensor:
        if self.block.is_parametric:
            block_mat = block_to_tensor(
                self.block, values, qubit_support=qubit_support, endianness=endianness  # type: ignore [arg-type]  # noqa
            ).squeeze(0)
        else:
            block_mat = self.block_mat

        return torch.sum(torch.matmul(state, block_mat) * state.conj(), dim=1)
