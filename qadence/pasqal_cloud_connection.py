from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from pasqal_cloud import SDK
from pasqal_cloud import Workload as WorkloadResult

from qadence import BackendName, QuantumModel


class ResultType(Enum):
    RUN = "run"
    SAMPLE = "sample"
    EXPECTATION = "expectation"


@dataclass(frozen=True)
class WorkloadSpec:
    model: QuantumModel
    result_types: list[ResultType]


class WorkloadType(Enum):
    # TODO Add other workload types supported by Qadence (Pulser, Emulator)
    QADENCE_CIRCUIT = "qadence_circuit"


@dataclass(frozen=True)
class WorkloadSpecJSON:
    workload_type: WorkloadType
    backend_type: BackendName
    config: str


def workload_spec_to_json(workload: WorkloadSpec) -> WorkloadSpecJSON:
    # TODO Implement this function correctly
    return WorkloadSpecJSON(WorkloadType.QADENCE_CIRCUIT, BackendName.PYQTORCH, "hello world!")


def upload_workload(connection: SDK, workload: WorkloadSpec) -> str:
    """Uploads a workload to Pasqal's Cloud and returns the created workload ID.

    Args:
        connection: A `pasqal_cloud.SDK` instance which is used to connect to the cloud.
        workload: A `WorkloadSpec` object, defining the specification of the workload that needs to
        be uploaded.

    Returns:
        A workload id as a `str`.
    """
    workload_json = workload_spec_to_json(workload)
    remote_workload = connection.create_workload(
        workload_json.workload_type, workload_json.backend_type, workload_json.config
    )
    workload_id: str = remote_workload.id
    return workload_id


class WorkloadNotDoneError(Exception):
    """Is raised if a workload is not yet finished running on remote."""

    pass


class WorkloadStoppedError(Exception):
    """Is raised when a workload has stopped running on remote for some reason."""

    pass


def check_status(connection: SDK, workload_id: str) -> WorkloadResult:
    """Checks if the workload is successfully finished on remote connection.

    Args:
        connection: A `pasqal_cloud.SDK` instance which is used to connect to the cloud.
        workload_id: the id `str` that is associated with the workload.

    Raises:
        WorkloadNotDoneError: Is raised when the workload status is "PENDING", "RUNNING" or
        "PAUSED".
        WorkloadStoppedError: Is raise when the workload status is "CANCELED", "TIMED_OUT" or
        "ERROR".
        ValueError: Is raised when the workload status has an unsupported value.

    Returns:
        The workload result if its status is "DONE" as a `pasqal_cloud.Workload` object.
    """
    # TODO Make the function return a "nice" result object
    result = connection.get_workload(workload_id)
    if result.status == "DONE":
        return result
    if result.status in ("PENDING", "RUNNING", "PAUSED"):
        raise WorkloadNotDoneError(
            f"Workload with id {workload_id} is not yet finished, the status is {result.status}"
        )
    if result.status in ("CANCELED", "TIMED_OUT", "ERROR"):
        message = f"Workload with id {workload_id} couldn't finish, the status is {result.status}."
        if result.status == "ERROR":
            message += f"The following error(s) occurred {result.errors}"
        raise WorkloadStoppedError(
            f"Workload with id {workload_id} couldn't finish, the status is {result.status}"
        )
    raise ValueError(
        f"Undefined workload status ({result.status}) was returned for workload ({result.id})"
    )


def get_result(
    connection: SDK, workload_id: str, timeout: float = 60.0, refresh_time: float = 1.0
) -> WorkloadResult:
    """Repeatedly checks if a workload has finished and returns the result.

    Args:
        connection: A `pasqal_cloud.SDK` instance which is used to connect to the cloud.
        workload_id: the id `str` that is associated with the workload.
        timeout: Time in seconds after which the function times out. Defaults to 60.0.
        refresh_time: Time in seconds after which the remote is requested to update the status
        again, when the workload is not finished yet. Defaults to 1.0.

    Raises:
        TimeoutError: _description_

    Returns:
        The workload result if its status is "DONE" as a `pasqal_cloud.Workload` object.
    """
    max_refresh_count = int(timeout // refresh_time)
    for _ in range(max_refresh_count):
        try:
            result = check_status(connection, workload_id)
        except WorkloadNotDoneError:
            time.sleep(refresh_time)
            continue
        return result
    raise TimeoutError("Request timed out because it wasn't finished in the specified time. ")
