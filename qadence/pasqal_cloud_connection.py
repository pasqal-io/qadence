from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pasqal_cloud import SDK
from pasqal_cloud import Workload as WorkloadResult
from torch import Tensor

from qadence import AbstractBlock, BackendName, QuantumCircuit, QuantumModel, serialize


class ResultType(Enum):
    RUN = "run"
    SAMPLE = "sample"
    EXPECTATION = "expectation"


@dataclass(frozen=True)
class WorkloadSpec:
    circuit: QuantumCircuit
    backend: BackendName
    result_types: list[ResultType]
    parameter_values: Optional[dict[str, Tensor]] = None
    observable: Optional[list[AbstractBlock]] = None


def get_spec_from_model(
    model: QuantumModel,
    result_types: list[ResultType],
    parameter_values: Optional[dict[str, Tensor]] = None,
    observable: Optional[list[AbstractBlock]] = None,
) -> WorkloadSpec:
    """Creates a `WorkloadSpec` from a quantum model.

    This function creates a `WorkloadSpec` from a `QuantumModel` and the other arguments provided.
    The circuit, that is extracted from the model, is the original circuit that was used to
    initialize the model, not the backend converted circuit in `model.circuit`. The backend set in
    the model will be used in the workload specification.

    It is important to note that in case there is an observable defined in the model, it is ignored
    in the workload specification. To provide an observable to the workload specification, it is
    only possible to set it in the observable argument of this function.

    Args:
        model: The quantum model that defines the circuit and backend for the workload spec.
        result_types: A list of result types that is requested in this workload.
        parameter_values: The parameter values that should be used during execution of the
        workload.
        observable: The observable to be used when the `ResultType.EXPECTATION` is requested.

    Returns:
        A `WorkloadSpec` instance based on the quantum model passed to this function.
    """
    circuit = model._circuit.original
    backend = model.backend
    return WorkloadSpec(circuit, backend, result_types, parameter_values, observable)


@dataclass(frozen=True)
class WorkloadSpecJSON:
    backend_type: str
    config: dict[str, str]
    workload_type = "qadence_circuit"


def _parameter_values_to_json(parameter_values: dict[str, Tensor]) -> str:
    result: dict[str, str] = dict()
    for key, value in parameter_values.items():
        result[key] = value.tolist()
    return json.dumps(result)


def workload_spec_to_json(workload: WorkloadSpec) -> WorkloadSpecJSON:
    """Serializes a `WorkloadSpec` into JSON format.

    Args:
        workload: A `WorkloadSpec` object, defining the specification of the workload that needs to
        be uploaded.

    Returns:
        Workload specification in JSON format.
    """
    circuit_json = json.dumps(serialize(workload.circuit))
    result_types_json = json.dumps([item.value for item in workload.result_types])
    config: dict[str, str] = {
        "circuit": circuit_json,
        "result_types": result_types_json,
    }

    if workload.parameter_values is not None:
        config["c_values"] = _parameter_values_to_json(workload.parameter_values)
    if workload.observable is not None:
        config["observable"] = json.dumps(
            [serialize(observable) for observable in workload.observable]
        )

    return WorkloadSpecJSON(str(workload.backend), config)


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
        raise WorkloadStoppedError(message)
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
