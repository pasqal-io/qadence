# Submission of Qadence Jobs to Pasqal Cloud

It is possible to submit quantum computational jobs to execute remotely on Pasqal's [cloud platform](https://portal.pasqal.cloud) from Qadence.
This feature can only be used if you have an account on the cloud platform, which has access to the Qadence workload.
The qadence module `qadence.pasqal_cloud_connection` offers functionality to specify the computation easily, upload the specification and retrieve the result when the computation has finished execution on the cloud platform.
In this tutorial, a simple quantum circuit will be defined as an example to showcase the submission process for remote computations.
The same process can be applied to run more complex quantum circuits on the cloud platform.

Let's first define a very simple quantum circuit that creates a Bell state.

```python
from qadence import CNOT, H, QuantumCircuit

circuit = QuantumCircuit(2, H(0), CNOT(0, 1))
```

If we want to upload this circuit to the cloud platform we need to follow 4 steps:
- Authentication and connection to cloud
- Defining workload specification
- Submission
- Retrieval of results

## Authentication and connection

To setup a connection the cloud platform, use the `SDK` object present in `qadence.pasqal_cloud_connection`. The email and password are the ones used to login to the webportal. The project-id can be found in the webportal under "Projects".

```python
from qadence.pasqal_cloud_connection import SDK

connection = SDK("john.doe@email.com", "my-password", project_id="proj1")
```
## Defining workload specification

To create a workload specification, we need some extra information on top the circuit itself.
We need to specify the backend, chosen here to be PyQTorch.
The cloud platform currently only supports PyQTorch.
Moreover, the requested result type needs to be defined.
Based on the workload specification, the appropriate run methods (`run`, `sample` or `expectation`) will be called by the `QuantumModel` by passing them through the enum value `ResultTypes` argument.
Moreover, the requested result type needs to be defined.
These are provided in a list, so that multiple result types can be requested in a single submission.

```python
from qadence import BackendName
from qadence.pasqal_cloud_connection import WorkloadSpec, ResultTypes

workload = WorkloadSpec(circuit, BackendName.PYQTORCH, [ResultTypes.SAMPLE, ResultTypes.RUN])
```

### Using a Quantum Model

If you already have your quantum computation defined as a `QuantumModel`, it is possible to create a workload specification directly from the model using `get_spec_from_model`.
Then, the circuit and backend specifications will be extracted from the model, the other values need to be provided as extra arguments.

```python
from qadence.pasqal_cloud_connection import get_spec_from_model

model = QuantumModel(...)
workload = get_spec_from_model(model, [ResultType.SAMPLE])
```

### Observable Expectation Value

For the result type `ResultType.EXPECTATION` it is mandatory to provide an observable to the workload specification.
In the example below we use the trivial identity observable `I(0) @ I(1)`.

```python
workload = WorkloadSpec(circuit, BackendName.PYQTORCH, [ResultTypes.EXPECTATION], observable=I(0)*I(1))
```

### Parametric Circuits

In the case of a parametric circuit, _i.e._ a circuit that contains feature parameters or variational parameters, values for these parameters need to be provided.
The parameter values are defined in a dictionary, where keys are the parameter name and values are parameter value passed as torch tensors.
The parameter values are defined in a dictionary, where keys are the parameter name and values are parameter value passed as torch tensors.
It is possible to set multiple values by using a 1-D tensor, to the parameters, in that case the computation is executed for each value in the tensor.
A mix of 0-D and 1-D tensors can be provided to keep some parameters constant and others changed during this process.
However, all 1-D tensors need to have the same length.


```python
parametric_circuit = ...
parameter_values = {"param1": tensor(0), "param2": tensor([0, 1, 2]), "param3": tensor([5, 6, 7])}
workload = WorkloadSpec(parametric_circuit, BackendName.PYQTORCH, [ResultTypes.EXPECTATION], parameter_values=parameter_values)
```

## Submission
Submission to the cloud platform is done very easily using the `submit_workload` function.
The workload id will be provided by executing the function.
This id is needed later, to request the status of the given workload.

```python
from qadence.pasqal_cloud_connection import submit_workload

workload_id = submit_workload(connection, workload)
```

## Check Workload Status

The `check_status` function can be used to see if the workload is finished already.
The status of a workload can be: done, pending, running, paused, canceled, timed out or error.
If so, the results of the computation will be provided in a `WorkloadResult` object.
The result of the computation itself can be found in the `result` attribute of this object.
If the workload has not finished yet, or resulted in an error, `check_status` will raise an exception, either a `WorkloadStoppedError` or `WorkloadNotDoneError`.

```python
from qadence.pasqal_cloud_connection import check_status

workload_result = check_status(connection, workload_id)
print(workload_result.result)
```

## Retrieval of Results

If you wish to wait for the workload to be finished, before moving further with your code, you can use the `get_result` function.
This function checks in set intervals the status of the workload until the workload is finished or the function has timed out.
The polling rate as well as the time out duration can be set optionally.

```python
from qadence.pasqal.cloud_connection import get_result

workload_result = get_result(connection, workload_id, timeout=60, refresh_time=1)
print(workload_result.result)
```
