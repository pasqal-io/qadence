A quantum program can be expressed and executed using the [`QuantumModel`][qadence.models.quantum_model.QuantumModel] type.
They serve three primary purposes:

_**Execution**_: by defining which backend the program is expected to be executed on. Qadence supports circuit compilation to the native backend representation.

_**Parameter handling**_: by conveniently handling and embedding the two parameter types that Qadence supports:
*feature* and *variational* (see more details in [this section](parameters.md)).

_**Differentiability**_: by enabling a *differentiable backend* that supports two differentiable modes: automated differentiation (AD) and parameter shift rule (PSR).
The former is used to differentiate non-gate parameters and enabled for PyTorch-based simulators only. The latter is used to differentiate gate parameters and is enabled for all backends.

!!! note "Backends"
    Quantum models can execute on a number of different purpose backends: simulators, emulators or real hardware.
    By default, Qadence executes on the [*PyQTorch*](https://github.com/pasqal-io/PyQ) backend which
    implements a state vector simulator. Other choices include the [*Pulser*](https://pulser.readthedocs.io/en/stable/)
    backend (pulse sequences on programmable neutral atom arrays).  For more information see
    [backend tutorial](backends.md).

The base `QuantumModel` exposes the following methods:

* `QuantumModel.run()`: To extract the wavefunction after circuit execution. Not supported by all backends.
* `QuantumModel.sample()`: Sample bitstring from the resulting quantum state after circuit execution. Supported by all backends.
* `QuantumModel.expectaction()`: Compute the expectation value of an observable.

Every `QuantumModel` is an instance of a
[`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) that enables differentiability for
its `expectation` method.

Upon construction of the model a compiled version of the abstract `QuantumCircuit` is
created:

```python exec="on" source="material-block" result="json" session="quantum-model"
from qadence import QuantumCircuit, QuantumModel, RX, Z, chain, Parameter, BackendName

# Construct a parametrized abstract circuit.
# At this point we cannot run anything yet.

x = Parameter("x")

n_qubits = 2
block = chain(RX(0, x), RX(1, x))
circuit = QuantumCircuit(n_qubits, block)
observable = Z(0)

# Construct a QuantumModel which will compile
# the abstract circuit to targetted backend.
model = QuantumModel(circuit, observable, backend=BackendName.PYQTORCH)

# now we construct a QuantumModel which will compile
# the abstract circuit to the backend we specify
model = QuantumModel(circuit, observable, backend="pyqtorch", diff_mode='ad')

# the converted circuit is a private attribute and should not
# manually be tampered with, but we can at least verify its there
print(model._circuit.native)

from pyqtorch.modules import QuantumCircuit as PyQCircuit
assert isinstance(model._circuit.native, PyQCircuit)
```

Now, the wavefunction, sample, or expectation value are computable:

```python exec="on" source="material-block" result="json" session="quantum-model"
import torch

# Set a batch of parameter values.
values = {"x": torch.rand(3)}

wf = model.run(values)
print(f"{wf=}")

xs = model.sample(values, n_shots=100)
print(f"{xs=}")

ex = model.expectation(values)
print(f"{ex=}")
```

You can also measure multiple observables by passing a list of blocks.
```python exec="on" source="material-block" result="json" session="quantum-model"
model = QuantumModel(circuit, [Z(0), Z(1)], backend="pyqtorch", diff_mode='ad')
ex = model.expectation(values)
print(ex)
```

### Quantum Neural Network (QNN)

The `QNN` is a subclass of the `QuantumModel` geared towards quantum machine learning and parameter optimisation. See the [ML
Tools](/tutorials/ml_tools.md) section or the [`QNN`][qadence.models.QNN] for more detailed
information and the [parametric program tutorial](/tutorials/parameters.md#parametrized-models) for parametrization.
