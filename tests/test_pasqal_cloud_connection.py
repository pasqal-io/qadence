from __future__ import annotations

from typing import Any, Callable
from unittest.mock import Mock

import pytest

from qadence.pasqal_cloud_connection import (
    QuantumModel,
    ResultType,
    WorkloadNotDoneError,
    WorkloadSpec,
    WorkloadStoppedError,
    check_status,
    get_result,
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
