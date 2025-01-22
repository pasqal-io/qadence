from __future__ import annotations

from typing import Any, Callable
from unittest.mock import Mock

import pytest
from torch import tensor

from qadence import BackendName, I, QuantumCircuit, QuantumModel
from qadence.pasqal_cloud_connection import (
    ResultType,
    WorkloadNotDoneError,
    WorkloadSpec,
    WorkloadStoppedError,
    _parameter_values_to_json,
    check_status,
    get_result,
    get_spec_from_model,
    submit_workload,
    _workload_spec_to_json,
)


def test_workload_spec_observable_validation(BasicQuantumCircuit: QuantumCircuit) -> None:
    with pytest.raises(ValueError):
        WorkloadSpec(BasicQuantumCircuit, BackendName.PYQTORCH, [ResultType.EXPECTATION])
    spec = WorkloadSpec(
        BasicQuantumCircuit, BackendName.PYQTORCH, [ResultType.EXPECTATION], observable=I(0)
    )
    assert spec.observable == I(0)


def test_workload_spec_parameter_values_validation(BasicQuantumCircuit: QuantumCircuit) -> None:
    with pytest.raises(ValueError):
        WorkloadSpec(
            BasicQuantumCircuit,
            BackendName.PYQTORCH,
            [ResultType.RUN],
            {"x": tensor([0, 2]), "y": tensor([0, 1, 2])},
        )
    with pytest.raises(ValueError):
        WorkloadSpec(
            BasicQuantumCircuit,
            BackendName.PYQTORCH,
            [ResultType.RUN],
            {"x": tensor([[0, 1], [2, 3]])},
        )
    WorkloadSpec(
        BasicQuantumCircuit,
        BackendName.PYQTORCH,
        [ResultType.RUN],
        {"x": tensor(0), "y": tensor([0, 1, 2])},
    )


def test_parameter_values_to_json() -> None:
    parameter_values = {"parameter1": tensor([0, 2]), "parameter2": tensor(2)}
    result = _parameter_values_to_json(parameter_values)
    expected = '{"parameter1": [0, 2], "parameter2": 2}'
    assert result == expected


def test_workload_spec_to_json_all_fields() -> None:
    circuit = QuantumCircuit(1, I(0))
    result_types = [ResultType.SAMPLE, ResultType.RUN, ResultType.EXPECTATION]
    workload = WorkloadSpec(
        circuit,
        BackendName.PYQTORCH,
        result_types,
        {"parameter": tensor([0, 1])},
        I(0),
    )
    result = _workload_spec_to_json(workload)
    assert result.workload_type == "qadence_circuit"
    assert result.backend_type == "pyqtorch"
    assert (
        result.config["circuit"]
        == '{"block": {"type": "I", "qubit_support": [0], "tag": null, "noise": null}, "register": '
        '{"graph": {"directed": false, "multigraph": false, "graph": {}, "nodes": [{"pos": [0.0, 0.'
        '0], "id": 0}], "links": []}, "device_specs": {"interaction": "NN", "rydberg_level": 60, "c'
        'oeff_xy": 3700.0, "max_detuning": 25.132741228718345, "max_amp": 18.84955592153876, "patte'
        'rn": {}, "type": "IdealDevice"}}}'
    )
    assert result.config["result_types"] == ["sample", "run", "expectation"]
    assert result.config["c_values"] == '{"parameter": [0, 1]}'
    assert (
        result.config["observable"]
        == '{"type": "I", "qubit_support": [0], "tag": null, "noise": null}'
    )


def test_workload_spec_to_json_no_optionals() -> None:
    workload = WorkloadSpec(
        QuantumCircuit(1, I(0)),
        BackendName.PYQTORCH,
        result_types=[
            ResultType.SAMPLE,
        ],
    )
    result = _workload_spec_to_json(workload)
    assert "observable" not in result.config.keys()
    assert result.config["c_values"] == "{}"


def test_get_spec_from_model(
    BasicQuantumModel: QuantumModel, BasicQuantumCircuit: QuantumCircuit
) -> None:
    workload = get_spec_from_model(BasicQuantumModel, [ResultType.SAMPLE])
    assert workload.circuit == BasicQuantumCircuit
    assert workload.backend == BackendName.PYQTORCH


def test_submit_workload(mocker: Any, BasicQuantumCircuit: QuantumCircuit) -> None:
    expected_workload_id = "my-workload"
    mock_connection_return = mocker.Mock()
    mock_connection_return.id = expected_workload_id
    mock_connection = mocker.Mock()
    mock_connection.create_workload.return_value = mock_connection_return
    circuit = BasicQuantumCircuit
    result_types = [ResultType.RUN, ResultType.SAMPLE]
    workload = WorkloadSpec(
        circuit, BackendName.PYQTORCH, result_types, {"my-parameter": tensor(3)}, I(0) * I(1)
    )
    result = submit_workload(mock_connection, workload)
    assert result == expected_workload_id


@pytest.fixture
def mock_status(mocker: Any) -> Callable:
    def generate_status(status: str) -> Mock:
        workload_result: Mock = mocker.Mock()
        workload_result.status = status
        return workload_result

    return generate_status


def test_check_status_done(mocker: Any, mock_status: Callable) -> None:
    mock_workload_result = mock_status("DONE")
    mock_connection = mocker.Mock()
    mock_connection.get_workload.return_value = mock_workload_result
    result = check_status(mock_connection, "my-workload")
    assert result is mock_workload_result


@pytest.mark.parametrize(
    "status,expected_error",
    [
        ("PENDING", WorkloadNotDoneError),
        ("RUNNING", WorkloadNotDoneError),
        ("PAUSED", WorkloadNotDoneError),
        ("CANCELED", WorkloadStoppedError),
        ("TIMED_OUT", WorkloadStoppedError),
        ("ERROR", WorkloadStoppedError),
        ("weird-status", ValueError),
    ],
)
def test_check_status_with_error(
    mocker: Any, status: str, expected_error: Exception, mock_status: Callable
) -> None:
    mock_workload_result = mock_status(status)
    mock_connection = mocker.Mock()
    mock_connection.get_workload.return_value = mock_workload_result
    with pytest.raises(expected_error):
        check_status(mock_connection, "my-workload")


def test_get_results(mocker: Any, mock_status: Callable) -> None:
    mocker.patch("qadence.pasqal_cloud_connection.time.sleep", return_value=None)
    mock_connection = mocker.Mock()
    result_done = mock_status("DONE")
    mock_connection.get_workload.side_effect = [
        mock_status("PENDING"),
        mock_status("PENDING"),
        mock_status("RUNNING"),
        result_done,
    ]
    result = get_result(mock_connection, "my-workload")
    assert result is result_done


def test_get_results_timeout(mocker: Any, mock_status: Callable) -> None:
    mocker.patch("qadence.pasqal_cloud_connection.time.sleep", return_value=None)
    connection = mocker.Mock()
    result_pending = mock_status("PENDING")
    connection.get_workload.side_effect = [
        result_pending,
        result_pending,
        result_pending,
        result_pending,
        result_pending,
    ]
    with pytest.raises(TimeoutError):
        get_result(connection, "my-workload", timeout=4)
