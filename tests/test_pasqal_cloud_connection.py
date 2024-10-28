from __future__ import annotations

from typing import Any

import pytest

from qadence.pasqal_cloud_connection import (
    QuantumModel,
    ResultType,
    WorkloadNotDoneError,
    WorkloadSpec,
    WorkloadStoppedError,
    check_status,
    upload_workload,
)


def test_upload_workload(mocker: Any, BasicQuantumModel: QuantumModel) -> None:
    expected_workload_id = "my-workload"
    mock_connection_return = mocker.Mock()
    mock_connection_return.id = expected_workload_id
    mock_connection = mocker.Mock()
    mock_connection.create_workload.return_value = mock_connection_return
    model = BasicQuantumModel
    result_types = [ResultType.RUN, ResultType.SAMPLE]
    workload = WorkloadSpec(model, result_types)
    result = upload_workload(mock_connection, workload)
    assert result == expected_workload_id


def test_check_status_done(mocker: Any) -> None:
    mock_workload_result = mocker.Mock()
    mock_workload_result.status = "DONE"
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
def test_check_status(mocker: Any, status: str, expected_error: Exception) -> None:
    mock_workload_result = mocker.Mock()
    mock_workload_result.status = status
    mock_connection = mocker.Mock()
    mock_connection.get_workload.return_value = mock_workload_result
    with pytest.raises(expected_error):
        check_status(mock_connection, "my-workload")
