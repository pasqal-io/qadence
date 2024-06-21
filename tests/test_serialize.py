from __future__ import annotations

from pathlib import Path

from sympy import *
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
    parse_expr_fn,
    save,
    serialize,
)


def test_serialization_parser() -> None:
    assert parse_expr_fn("Float('-0.33261030434342942', precision=53)") is True
    assert parse_expr_fn("Mul(Float('2.0',precision=53), Parameter('theta_0'))") is True
    assert parse_expr_fn("2*3") is False
    assert parse_expr_fn("__import__('os').system('ls -la')") is False


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
