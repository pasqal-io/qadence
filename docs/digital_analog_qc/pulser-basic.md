Qadence offers a direct interface with Pulser[^1], an open-source pulse-level interface written in Python and
specifically designed for programming neutral atom quantum computers.

Using directly Pulser requires deep knowledge on pulse-level programming and on how neutral atom devices work. Qadence abstracts this complexity out by using the familiar block-based interface for building pulse sequences in Pulser while leaving the possibility
to directly manipulate them if required by, for instance, optimal pulse shaping.

!!! note
    The Pulser backend is still experimental and the interface might change in the future.

Let's see it in action.

## Default qubit interaction

When simulating pulse sequences written using Pulser, the underlying Hamiltonian it
constructs is equivalent to a digital-analog quantum computing program with the following interaction
Hamiltonian (see [digital-analog emulation](analog-basics.md) for more details):

$$
\mathcal{H}_{int} = \sum_{i<j} \frac{C_6}{|R_i - R_j|^6} \hat{n}_i \hat{n}_j
$$

where $C_6$ is an interaction coefficient which depends on the principal quantum number of chosen
the neutral atom system, $R_i$ are the atomic positions in Cartesian coordinates
and $\hat{n} = \frac{1-\sigma^z_i}{2}$ is the number operator.

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
| `wait`      | An idle block to wait for the system to evolve for a specific time according to the interaction. | free evolution time |

## Two qubits register: Bell state

Using the `chain` block makes it easy to create a gate sequence. Here is an example of how to create a Bell state.
The `entangle` operation uses `CZ` interactions (according to the interaction Hamiltonian introduced in the first paragraph of this section)
to entangle states on the `X` basis. We move the qubits back to
the `Z` basis for the readout using a `Y` rotation.

```python exec="on" source="material-block" session="pulser-basic"
from qadence import chain, entangle, RY

bell_state = chain(
   entangle("t", qubit_support=(0,1)),
   RY(0, "y"),
)
```

To convert the chain block into a pulse sequence, we define a `Register` with two qubits and combine it to create a circuit as usual. Then we construct a `QuantumModel` with a Pulser backend to convert it into a proper parametrized pulse sequence. Supplying the
parameter values allows to sample from the pulse sequence result like any other Qadence backend.

```python exec="on" source="material-block" "html=1" session="pulser-basic"
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

# return the final state vector
final_vector = model.run(params)
print(final_vector)

# sample from the result state vector and plot the distribution
sample = model.sample(params, n_shots=50)[0]
print(sample)

fig, ax = plt.subplots()
ax.bar(sample.keys(), sample.values())
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

At variance with other backends, the Pulser one provides the concept of `Device`, inherited from
the Pulser. Check [this](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html) tutorial for more
information.

A `Device` instance encapsulates all the properties for the definition of a real neutral atoms processor, including but not limited
to the maximum laser amplitude for the pulses, the maximum distance between two qubits and the maximum duration of the pulse.

Qadence offers a simplified interface with only two devices which are detailed [here](../backends/pulser.md):

* `IDEALIZED` (default): ideal device which should be used only for testing purposes. It does not have any limitation
in what pulse sequences can run with it.
* `REALISTIC`: device specification very similar to a real neutral atom quantum processor.

!!! note
    If you want to perform simulations closer to the specifications of real neutral atom machines,
    always select the `REALISTIC` device.

One can use the `Configuration` of the Pulser backend to select the appropriate device:

```python exec="on" source="material-block" session="pulser-basic"
from qadence.backends.pulser.devices import Device

register = Register(2)
circuit = QuantumCircuit(register, bell_state)

# choose a realistic device
model = QuantumModel(
    circuit,
    backend="pulser",
    diff_mode="gpsr",
    configuration={"device_type": Device.REALISTIC}
)

params = {
    "t": torch.tensor([383]),  # ns
    "y": torch.tensor([3*torch.pi/2]),
}

# sample from the result state vector and plot the distribution
sample = model.sample(params, n_shots=50)[0]
print(sample)
```

## Create your own gate

A big advantage of using the block-based interface if  Qadence is that it makes it easy to create complex
operations from simple ones as a block composition. In the following, we use the entanglement operation as an example.

The operation consists of moving _all_ the qubits to the `X` basis having the atoms' interaction perform a controlled-Z operation
during the free evolution. And we can easily recreate this pattern using the `wait` (corresponding to free evolution) and `AnalogRY` blocks with appropriate parameters.

```python exec="on" source="material-block" session="pulser-basic"
from qadence import AnalogRY, chain, wait

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
model = QuantumModel(circuit, backend="pulser", diff_mode='gpsr')

params = {
    "t": torch.tensor([383]),  # ns
    "y": torch.tensor([torch.pi / 2]),
}

sample = model.sample(params, n_shots=50)[0]

fig, ax = plt.subplots()
plt.bar(sample.keys(), sample.values())
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
model = QuantumModel(circuit, backend="pulser", diff_mode="gpsr")

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

model.assign_parameters(params).draw(draw_phase_area=True, show=True)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## References

[^1]: [Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays](https://pulser.readthedocs.io/en/stable/)
