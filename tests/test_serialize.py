from __future__ import annotations

from pathlib import Path
from typing import Any

from sympy import Expr
from torch import isclose

from qadence import QuantumCircuit
from qadence.blocks import AbstractBlock, KronBlock
from qadence.ml_tools.models import TransformedModule
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN, QuantumModel
from qadence.register import Register
from qadence.serialization import (
    FORMAT_DICT,
    SerializationFormat,
    deserialize,
    load,
    save,
    serialize,
)


def test_non_module_serialization(
    tmp_path: Path,
    BasicQuantumCircuit: QuantumCircuit,
    BasicExpression: Expr,
    BasicRegister: Register,
    BasicFeatureMap: KronBlock,
    BasicObservable: AbstractBlock,
) -> None:
    for obj in [
        BasicQuantumCircuit,
        BasicFeatureMap,
        BasicExpression,
        BasicRegister,
        BasicObservable,
    ]:
        assert obj == deserialize(serialize(obj))
        save(obj, tmp_path, "obj")
        loaded_obj = load(tmp_path / Path("obj.json"))
        assert obj == loaded_obj


def test_qm_serialization(tmp_path: Path, BasicQuantumModel: QuantumModel) -> None:
    _m = BasicQuantumModel
    inputs = rand_featureparameters(_m, 1)
    for save_params in [True, False]:
        exp = _m.expectation(inputs)
        d = serialize(_m, save_params)
        qm_ser = deserialize(d, save_params)  # type: ignore[assignment]
        exp_ser = qm_ser.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_ser)  # type: ignore[union-attr]
    for FORMAT in SerializationFormat:
        save(_m, tmp_path, "obj", FORMAT)
        suffix, _, _, _ = FORMAT_DICT[FORMAT]
        qm = load(tmp_path / Path("obj" + suffix))
        exp_l = qm.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_l)


def test_qnn_serialization(tmp_path: Path, BasicQNN: QNN) -> None:
    _m = BasicQNN
    inputs = rand_featureparameters(_m, 1)
    for save_params in [True, False]:
        exp = _m.expectation(inputs)
        d = serialize(_m, save_params)
        qm_ser = deserialize(d, save_params)  # type: ignore[assignment]
        exp_ser = qm_ser.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_ser)  # type: ignore[union-attr]
    for FORMAT in SerializationFormat:
        save(_m, tmp_path, "obj", FORMAT)
        suffix, _, _, _ = FORMAT_DICT[FORMAT]
        qm = load(tmp_path / Path("obj" + suffix))
        exp_l = qm.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_l)


def test_tm_serialization(tmp_path: Path, BasicTransformedModule: TransformedModule) -> None:
    _m = BasicTransformedModule
    inputs = rand_featureparameters(_m, 1)
    for save_params in [True, False]:
        exp = _m.expectation(inputs)
        d = serialize(_m, save_params)
        qm_ser = deserialize(d, save_params)  # type: ignore[assignment]
        exp_ser = qm_ser.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_ser)  # type: ignore[union-attr]
    for FORMAT in SerializationFormat:
        save(_m, tmp_path, "obj", FORMAT)
        suffix, _, _, _ = FORMAT_DICT[FORMAT]
        qm = load(tmp_path / Path("obj" + suffix))
        exp_l = qm.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_l)


def test_external_serialization(
    tmp_path: Path, BasicQuantumCircuit: QuantumCircuit, BasicObservable: AbstractBlock
) -> None:
    class ExternalModel(QuantumModel):
        def __init__(
            self,
            circuit: QuantumCircuit,
            observable: AbstractBlock,
            **qm_kwargs: Any,
        ) -> None:
            super().__init__(circuit, observable, **qm_kwargs)

    def deserialize_fn(d: dict) -> ExternalModel:
        return ExternalModel._from_dict(d)

    _m = ExternalModel(BasicQuantumCircuit, BasicObservable)
    inputs = rand_featureparameters(_m, 1)

    exp = _m.expectation(inputs)
    d = serialize(_m)
    qm_ser = deserialize_fn(d)
    exp_ser = qm_ser.expectation(inputs)
    assert isclose(exp, exp_ser)

    for FORMAT in SerializationFormat:
        save(_m, tmp_path, "obj", FORMAT)
        suffix, _, _, _ = FORMAT_DICT[FORMAT]
        qm = load(tmp_path / Path("obj" + suffix), deserialize_fn=deserialize_fn)
        exp_l = qm.expectation(inputs)  # type: ignore[union-attr]
        assert isclose(exp, exp_l)
