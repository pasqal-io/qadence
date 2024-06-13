from __future__ import annotations

import os
from dataclasses import dataclass
from functools import singledispatch
from typing import Any
from uuid import uuid4

from torch import Tensor, rand

from qadence.blocks import AbstractBlock, parameters
from qadence.circuit import QuantumCircuit
from qadence.logger import get_script_logger
from qadence.ml_tools.models import TransformedModule
from qadence.models import QNN, QuantumModel
from qadence.parameters import Parameter, stringify

logger = get_script_logger(__name__)


@singledispatch
def rand_featureparameters(
    x: QuantumCircuit | AbstractBlock | QuantumModel | QNN | TransformedModule, *args: Any
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


@rand_featureparameters.register
def _(tm: TransformedModule, batch_size: int = 1) -> dict[str, Tensor]:
    return rand_featureparameters(tm.model, batch_size)


@dataclass
class MLFlowConfig:
    """
    Example:

        export MLFLOW_TRACKING_URI=tracking_uri
        export MLFLOW_TRACKING_USERNAME=username
        export MLFLOW_TRACKING_PASSWORD=password
    """

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    MLFLOW_TRACKING_USERNAME: str = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    MLFLOW_TRACKING_PASSWORD: str = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", str(uuid4()))

    def __post_init__(self) -> None:
        if self.MLFLOW_TRACKING_USERNAME != "":
            logger.info(
                f"Intialized mlflow remote logging for user {self.MLFLOW_TRACKING_USERNAME}."
            )
