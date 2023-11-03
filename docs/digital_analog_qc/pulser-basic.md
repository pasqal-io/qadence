Qadence offers a direct interface with Pulser[^1], an open-source pulse-level interface written in Python and
specifically designed for programming neutral atom quantum computers.

Using directly Pulser requires advanced knowledge on pulse-level programming and on how neutral atom devices work. Qadence abstracts this complexity out by using the familiar block-based interface for building pulse sequences in Pulser while leaving the possibility
to directly manipulate them if required by, for instance, optimal pulse shaping.

!!! note
    The Pulser backend is still experimental and the interface might change in the future.
    Please note that it does not support `DiffMode.AD`.

!!! note
    With the Pulser backend, `qadence` simulations can be executed on the cloud emulators available on the PASQAL
    cloud platform. In order to do so, make to have valid credentials for the PASQAL cloud platform and use
    the following configuration for the Pulser backend:

    ```python exec="off" source="material-block" html="1" session="pulser-basic"
    config = {
        "cloud_configuration": {
            "username": "<changeme>",
            "password": "<changeme>",
            "project_id": "<changeme>",  # the project should have access to emulators
            "platform": "EMU_FREE"  # choose between `EMU_TN` and `EMU_FREE`
        }
    }
    ```

For inquiries and more details on the cloud credentials, please contact
[info@pasqal.com](mailto:info@pasqal.com).


## Default qubit interaction

When simulating pulse sequences written using Pulser, the underlying constructed Hamiltonian
is equivalent to a digital-analog quantum computing program (see [digital-analog emulation](analog-basics.md) for more details)
with the following interaction term:

$$
\mathcal{H}_{\textrm{int}} = \sum_{i<j} \frac{C_6}{|R_i - R_j|^6} \hat{n}_i \hat{n}_j
$$

where $C_6$ is an interaction strength coefficient dependent on the principal quantum number of chosen
the neutral atom system, $R_i$ are atomic positions in Cartesian coordinates
and $\hat{n} = \frac{1-\sigma^z_i}{2}$ the number operator.

!!! note
    The Ising interaction is **always-on** for all computations performed with the
    Pulser backend. It cannot be switched off.

## Available quantum operations

Currently, the Pulser backend supports the following operations:

| gate        | description                                                                                      | trainable parameter |
|-------------|--------------------------------------------------------------------------------------------------|---------------------|
| `RX`, `RY`     | Single qubit rotations. Notice that the interaction is on and this affects the resulting gate fidelity.                                                                        | rotation angle      |
| `AnalogRX`, `AnalogRY`, `AnalogRZ` | Span a single qubit rotation among the entire register.                                          | rotation angle      |
| `entangle`  | Fully entangle the register.                                                                     | interaction time    |
| `wait`      | An idle block to wait for the system to free-evolve for a duration according to the interaction. | free evolution time |

## Sequence the Bell state on a two qubit register

The next example illustrates how to create a pulse sequence to prepare a Bell state. This is a sequence of an entanglement operation,
represented as an `entangle` gate (using `CZ` interactions) in the $X$-basis and a $Y$ rotation for readout
in the $Z$-basis:

```python exec="on" source="material-block" result="json" session="pulser-basic"
from qadence import chain, entangle, RY

bell_state = chain(
   entangle("t", qubit_support=(0,1)),
   RY(0, "y"),
)
print(f"bell_state = {bell_state}") # markdown-exec: hide
```

Next, a `Register` with two qubits is combined with the resulting `ChainBlock` to form a circuit.
Then, the `QuantumModel` converts the circuit into a proper parametrized
pulse sequence with the Pulser backend. Supplying the parameter values allows to sample the pulse sequence outcome:

```python exec="on" source="material-block" result="json" session="pulser-basic"
import torch
import matplotlib.pyplot as plt
from qadence import Register, QuantumCircuit, QuantumModel

register = Register(2)
circuit = QuantumCircuit(register, bell_state)
model = QuantumModel(circuit, backend="pulser", diff_mode="gpsr")

params = {
    "t": torch.tensor([383]),  # ns
    "y": torch.tensor([3*torch.pi/2]),
}

# Return the final state vector
final_vector = model.run(params)
print(f"final_vector = {final_vector}") # markdown-exec: hide

# Sample from the result state vector
sample = model.sample(params, n_shots=50)[0]
print(f"sample = {sample}") # markdown-exec: hide
```

Plot the distribution:

```python exec="on" source="material-block" html="1" session="pulser-basic"
fig, ax = plt.subplots() # markdown-exec: hide
plt.xlabel("Bitstring") # markdown-exec: hide
plt.ylabel("Counts") # markdown-exec: hide
ax.bar(sample.keys(), sample.values()) # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```
One can visualise the pulse sequence with different parameters using the `assign_paramters` method.

```python exec="on" source="material-block" html="1" session="pulser-basic"
model.assign_parameters(params).draw(show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Change device specifications

At variance with other backends, Pulser provides the concept of `Device`. A `Device` instance encapsulates
all the properties for the definition of a real neutral atoms processor, including but not limited
to the maximum laser amplitude for pulses, the maximum distance between two qubits and the maximum duration
of the pulse. For more information, please check this [tutorial](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html).

Qadence offers a simplified interface with only two devices which are detailed [here](../backends/pulser.md):

* `IDEALIZED` (default): ideal device which should be used only for testing purposes. It does not restrict the
simulation of pulse sequences.
* `REALISTIC`: device specification close to real neutral atom quantum processors.

!!! note
    If you want to perform simulations closer to the specifications of real neutral atom machines,
    always select the `REALISTIC` device.

One can use the `Configuration` of the Pulser backend to select the appropriate device:

```python exec="on" source="material-block" result="json" session="pulser-basic"
from qadence import BackendName, DiffMode
from qadence.backends.pulser.devices import Device

register = Register(2)
circuit = QuantumCircuit(register, bell_state)

# Choose a realistic device
model = QuantumModel(
    circuit,
    backend=BackendName.PULSER,
	diff_mode=DiffMode.GPSR,
    configuration={"device_type": Device.REALISTIC}
)

params = {
    "t": torch.tensor([383]),  # ns
    "y": torch.tensor([3*torch.pi/2]),
}

# Sample from the result state vector
sample = model.sample(params, n_shots=50)[0]
print(f"sample = {sample}") # markdown-exec: hide
```

## Create a custom gate

A major advantage of the block-based interface in Qadence is the ease to compose complex
operations from a restricted set of primitive ones. In the following, a custom entanglement operation is used as an example.

The operation consists of moving _all_ the qubits to the $X$-basis. This is realized when the atomic interaction performs a
controlled-$Z$ operation during the free evolution. As seen before, this is implemented with the `wait` and `AnalogRY` blocks and
appropriate parameters.

```python exec="on" source="material-block" session="pulser-basic"
from qadence import AnalogRY, chain, wait

# Custom entanglement operation.
def my_entanglement(duration):
    return chain(
        AnalogRY(-torch.pi / 2),
        wait(duration)
    )

protocol = chain(
   my_entanglement("t"),
   RY(0, "y"),
)

register = Register(2)
circuit = QuantumCircuit(register, protocol)
model = QuantumModel(circuit, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR)

params = {
    "t": torch.tensor([383]),  # ns
    "y": torch.tensor([torch.pi / 2]),
}

sample = model.sample(params, n_shots=50)[0]

fig, ax = plt.subplots() # markdown-exec: hide
plt.xlabel("Bitstring") # markdown-exec: hide
plt.ylabel("Counts") # markdown-exec: hide
plt.bar(sample.keys(), sample.values()) # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

## Digital-analog QNN circuit

Finally, let's put all together by constructing a digital-analog
version of a quantum neural network circuit with feature map and variational
ansatz.

```python exec="on" source="material-block" html="1" session="pulser-basic"
from qadence import kron, fourier_feature_map
from qadence.operations import RX, RY, AnalogRX

hea_one_layer = chain(
    kron(RY(0, "th00"), RY(1, "th01")),
    kron(RX(0, "th10"), RX(1, "th11")),
    kron(RY(0, "th20"), RY(1, "th21")),
    entangle("t", qubit_support=(0,1)),
)

protocol = chain(
    fourier_feature_map(1, param="x"),
    hea_one_layer,
    AnalogRX(torch.pi/4)
)

register = Register(2)
circuit = QuantumCircuit(register, protocol)
model = QuantumModel(circuit, backend=BackendName.PULSER, diff_mode=DiffMode.GPSR)

params = {
    "x": torch.tensor([0.8]), # rad
    "t": torch.tensor([900]), # ns
    "th00":  torch.rand(1), # rad
    "th01":  torch.rand(1), # rad
    "th10":  torch.rand(1), # rad
    "th11":  torch.rand(1), # rad
    "th20":  torch.rand(1), # rad
    "th21":  torch.rand(1), # rad
}

model.assign_parameters(params).draw(draw_phase_area=True, show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## References

[^1]: [Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays](https://pulser.readthedocs.io/en/stable/)
