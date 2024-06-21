from __future__ import annotations

from functools import singledispatch
from typing import Any

from torch import Tensor, rand

from qadence import QNN, QuantumModel
from qadence.blocks import AbstractBlock, parameters
from qadence.circuit import QuantumCircuit
from qadence.parameters import Parameter, stringify


@singledispatch
def rand_featureparameters(
    x: QuantumCircuit | AbstractBlock | QuantumModel | QNN, *args: Any
) -> dict[str, Tensor]:
    raise NotImplementedError(f"Unable to generate random featureparameters for object {type(x)}.")


@rand_featureparameters.register
def _(block: AbstractBlock, batch_size: int = 1) -> dict[str, Tensor]:
    non_number_params = [p for p in parameters(block) if not p.is_number]
    feat_params: list[Parameter] = [p for p in non_number_params if not p.trainable]
    return {stringify(p): rand(batch_size, requires_grad=False) for p in feat_params}


@rand_featureparameters.register
def _(circuit: QuantumCircuit, batch_size: int = 1) -> dict[str, Tensor]:
    return rand_featureparameters(circuit.block, batch_size)


@rand_featureparameters.register
def _(qm: QuantumModel, batch_size: int = 1) -> dict[str, Tensor]:
    return rand_featureparameters(qm._circuit.abstract, batch_size)


@rand_featureparameters.register
def _(qnn: QNN, batch_size: int = 1) -> dict[str, Tensor]:
    return rand_featureparameters(qnn._circuit.abstract, batch_size)
