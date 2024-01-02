from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import Parameter as TorchParam

from qadence import (
    BackendName,
    DiffMode,
    Parameter,
    QuantumCircuit,
    deserialize,
    load,
    save,
    serialize,
)
from qadence.blocks import chain, tag
from qadence.constructors import hamiltonian_factory, hea
from qadence.ml_tools.models import TransformedModule
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN
from qadence.operations import RY, Z
from qadence.serialization import SerializationFormat

np.random.seed(42)
torch.manual_seed(42)


def quantum_circuit(n_qubits: int = 2, depth: int = 1) -> QuantumCircuit:
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    tag(fm, "feature_map")

    ansatz = hea(n_qubits=n_qubits, depth=depth)
    tag(ansatz, "ansatz")

    return QuantumCircuit(n_qubits, fm, ansatz)


def get_qnn(n_qubits: int, depth: int) -> QNN:
    observable = hamiltonian_factory(n_qubits, detuning=Z)
    circuit = quantum_circuit(n_qubits=n_qubits, depth=depth)
    model = QNN(circuit, observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)
    init_params = torch.rand(model.num_vparams)
    model.reset_vparams(init_params)
    return model


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_transformed_module(n_qubits: int) -> None:
    depth = 1
    model = get_qnn(n_qubits, depth)
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)
    assert not torch.isnan(pred)

    transformed_model = TransformedModule(
        model,
        None,
        None,
        TorchParam(torch.tensor(5.0)),
        2.0,
        1000.0,
        TorchParam(torch.tensor(10.0)),
    )
    pred_transformed = transformed_model(input_values)
    assert not torch.isnan(pred_transformed)


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_same_output(n_qubits: int) -> None:
    depth = 1
    model = get_qnn(n_qubits, depth)
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)
    assert not torch.isnan(pred)

    transformed_model = TransformedModule(
        model,
        None,
        None,
        TorchParam(torch.tensor(1.0)),
        0.0,
        1.0,
        TorchParam(torch.tensor(0.0)),
    )
    pred_transformed = transformed_model(input_values)
    assert torch.allclose(pred_transformed.real, pred)
    assert pred.size() == pred_transformed.size()


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_no_scaling_provided(n_qubits: int) -> None:
    depth = 1
    model = get_qnn(n_qubits, depth)
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)
    assert not torch.isnan(pred)

    transformed_model = TransformedModule(model, None, None, None, 2.0, None, 100.0)
    pred_transformed = transformed_model(input_values)
    assert not torch.isnan(pred_transformed)
    assert pred.size() == pred_transformed.size()


@pytest.mark.parametrize("n_qubits", [2, 4, 8])
def test_no_args(n_qubits: int) -> None:
    depth = 1
    model = get_qnn(n_qubits, depth)
    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    pred = model(input_values)
    assert not torch.isnan(pred)

    transformed_model = TransformedModule(model)
    pred_transformed = transformed_model(input_values)
    assert torch.allclose(pred_transformed.real, pred)
    assert pred.size() == pred_transformed.size()


def test_save_load_TM_pyq(tmp_path: Path, BasicTransformedModule: TransformedModule) -> None:
    tm = BasicTransformedModule
    # tm._input_scaling = torch.rand(1)
    # tm._input_shifting = torch.rand(1)
    # tm._output_scaling = torch.rand(1)
    # tm._output_shifting = torch.rand(1)
    # serialize deserialize
    d = serialize(tm)
    tm_ser = deserialize(d)  # type: ignore[assignment]
    inputs = rand_featureparameters(tm, 1)
    y_p0 = tm(inputs)[0]
    y_p1 = tm_ser(inputs)[0]  # type: ignore[operator]
    assert torch.allclose(y_p0, y_p1)
    # save load
    for _format, _suffix in zip(
        [SerializationFormat.JSON, SerializationFormat.PT], [".json", ".pt"]
    ):
        base_name = "tm"
        save(tm, tmp_path, base_name, _format)
        tm_load = load(tmp_path / (base_name + _suffix))  # type: ignore[assignment]
        y_px = tm_load.expectation(inputs)[0]  # type: ignore[union-attr]
        assert tm.in_features == tm_load.in_features  # type: ignore[union-attr]
        assert tm.out_features == tm_load.out_features  # type: ignore[union-attr]
        assert tm._input_scaling == tm_load._input_scaling  # type: ignore[union-attr]
        assert tm._input_shifting == tm_load._input_shifting  # type: ignore[union-attr]
        assert tm._output_scaling == tm_load._output_scaling  # type: ignore[union-attr]
        assert tm._output_shifting == tm_load._output_shifting  # type: ignore[union-attr]

        assert torch.allclose(y_p0, y_px)


def test_basic_save_load_ckpts(Basic: torch.nn.Module, tmp_path: Path) -> None:
    model = Basic
    in_feat = 1
    x = torch.rand(in_feat)
    exp_no = model(x)
    tm = TransformedModule(
        model=model,
        in_features=in_feat,
        out_features=1,
        input_scaling=torch.ones(in_feat),
        input_shifting=torch.zeros(in_feat),
        output_scaling=torch.ones(1),
        output_shifting=torch.zeros(1),
    )
    exp_tm = tm(x)
    assert exp_no.shape == exp_tm.shape
    assert torch.allclose(exp_no, exp_tm)
