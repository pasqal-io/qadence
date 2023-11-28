

## Physics behind semi-local addressing patterns

Recall that in Qadence the general neutral-atom Hamiltonian for a set of $n$ interacting qubits is given by expression

$$
\mathcal{H} = \mathcal{H}_{\rm drive} + \mathcal{H}_{\rm int} = \sum_{i=0}^{n-1}\left(\mathcal{H}^\text{d}_{i}(t) + \sum_{j<i}\mathcal{H}^\text{int}_{ij}\right)
$$

as is described in detail in the [analog interface basics](analog-basics.md) documentation.

The driving Hamiltonian term in priciple can model any local single-qubit rotation by addressing each qubit individually. However, on the current generation of neutral-atom devices full local addressing is not yet achievable. Fortunatelly, using devices called spatial light modulators (SLMs) it is possible to implement semi-local addressing where the pattern of targeted qubits is fixed during the execution of the quantum circuit. More formally, the addressing pattern appears as an additional term in the neutral-atom Hamiltonian:

$$
\mathcal{H} = \mathcal{H}_{\rm drive} + \mathcal{H}_{\rm int} + \mathcal{H}_{\rm pattern},
$$

where $\mathcal{H}_{\rm pattern}$ is given by

$$
\mathcal{H}_{\rm pattern} = \sum_{i=0}^{n-1}\left(-\Delta w_i^{\rm det} N_i + \Gamma w_i^{\rm drive} X_i\right).
$$

Here $\Delta$ specifies the maximal **negative** detuning that each qubit in the register can be exposed to. The weight $w_i^{\rm det}\in [0, 1]$ determines the actual value of detuning that $i$-th qubit feels and this way the detuning pattern is emulated. Similarly, for the amplitude pattern $\Gamma$ determines the maximal additional **positive** drive that acts on qubits. In this case the corresponding weights $w_i^{\rm drive}$ can vary in the interval $[0, 1]$.

Using the detuning and amplitude patterns described above one can modify the behavior of a selected set of qubits, thus achieving semi-local addressing.

## Creating semi-local addressing patterns

In Qadence semi-local addressing patterns can be created by either specifying fixed values for the weights of qubits to be addressed or defining them as trainable parameters that can be optimized later in some training loop. Semi-local addressing patterns can be defined with the `AddressingPattern` dataclass.

### Fixed weights

With fixed weights detuning/amplitude addressing patterns can defined in the following way.

```python exec="on" source="material-block" session="emu"
import torch
from qadence.analog import AddressingPattern

n_qubits = 3

w_det = {0: 0.9, 1: 0.5, 2: 1.0}
w_amp = {0: 0.1, 1: 0.4, 2: 0.8}
det = 9.0
amp = 6.5
pattern = AddressingPattern(
    n_qubits=n_qubits,
    det=det,
    amp=amp,
    weights_det=w_det,
    weights_amp=w_amp,
)
```

If only detuning or amplitude pattern is needed - the corresponding weights for all qubits can be set to 0.

The created addressing pattern can now be passed as a characteristic of the `RydbergDevice`, defining the device specifications of a given `Register`.

```python exec="on" source="material-block" session="emu"
from qadence import (
    AnalogRX,
    AnalogRY,
    BackendName,
    DiffMode,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    Register,
    chain,
    total_magnetization,
    IdealDevice,
)

# define register and circuit
spacing = 8.0
x = Parameter("x")
block = chain(AnalogRX(3 * x), AnalogRY(0.5 * x))

device_specs = IdealDevice(pattern = pattern)

reg = Register.line(
    n_qubits,
    spacing=spacing,
    device_specs=device_specs,
)

circ = QuantumCircuit(reg, block)

# define parameter values and observable
values = {"x": torch.linspace(0.5, 2 * torch.pi, 50)}
obs = total_magnetization(n_qubits)

model_pyq = QuantumModel(
    circuit=circ, observable=obs, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
)

# calculate expectation value of the circuit
expval_pyq = model_pyq.expectation(values=values)
```

The same configuration can also be seamlessly used to create a model on the Pulser backend.

```python exec="on" source="material-block" session="emu"
model_pulser = QuantumModel(
    circuit=circ,
    observable=obs,
    backend=BackendName.PULSER,
    diff_mode=DiffMode.GPSR
)

# calculate expectation value of the circuit
expval_pulser = model_pulser.expectation(values=values)
```

### Trainable weights

Since the user can specify both the maximal detuning/amplitude value of the addressing pattern and the corresponding weights, it is natural to make these parameters variational in order to use them in some QML setting. This can be achieved by defining pattern weights as trainable `Parameter` instances or strings specifying weight names.

```python exec="on" source="material-block" session="emu"
n_qubits = 3
reg = Register.line(n_qubits, spacing=8.0)

# some random target function value
f_value = torch.rand(1)

# define trainable addressing pattern
w_amp = {i: f"w_amp{i}" for i in range(n_qubits)}
w_det = {i: f"w_det{i}" for i in range(n_qubits)}
amp = "amp"
det = "det"

pattern = AddressingPattern(
    n_qubits=n_qubits,
    det=det,
    amp=amp,
    weights_det=w_det,
    weights_amp=w_amp,
)

# some fixed analog operation
block = AnalogRX(torch.pi)

device_specs = IdealDevice(pattern = pattern)

reg = Register.line(
    n_qubits,
    spacing=spacing,
    device_specs=device_specs,
)

circ = QuantumCircuit(reg, block)

# define quantum model
obs = total_magnetization(n_qubits)
model = QuantumModel(circuit=circ, observable=obs, backend=BackendName.PYQTORCH)

# prepare for training
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_criterion = torch.nn.MSELoss()
n_epochs = 200
loss_save = []

# train model
for _ in range(n_epochs):
    optimizer.zero_grad()
    out = model.expectation()
    loss = loss_criterion(f_value, out)
    loss.backward()
    optimizer.step()
    loss_save.append(loss.item())

# get final results
f_value_model = model.expectation().detach()

assert torch.isclose(f_value, f_value_model, atol=0.01)

print("The target function value: ", f_value)
print("The trained function value: ", f_value_model)
```

Here the value of expectation of the circuit is fitted to some predefined value by varying the parameters of the addressing pattern.

!!! note
    Trainable parameters currently are supported only by `pyqtorch` backend.
