Qadence offers a direct interface with Pulser[^1], a pulse-level interface specifically designed for programming neutral atom quantum computers.

Using directly Pulser requires deep knowledge on pulse-level programming and on how neutral atom devices work. Qadence abstracts out this complexity by using the familiar block-based interface for building pulse sequences in Pulser while leaving the possibility
to directly manipulate them if required.

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

where $C_6$ is an interaction coefficient which depends on the principal quantum number of
the neutral atom system, $R_i$ are the atomic position in Cartesian coordinates
and $\hat{n} = \frac{1-\sigma^z_i}{2}$ is the number operator.

Notice that this interaction is **always-on** for any computation performed with the Pulser backend and cannot be switched off.

## Pulse sequences with Qadence

Currently, the backend supports the following operations:

| gate        | description                                                                                      | trainable parameter |
|-------------|--------------------------------------------------------------------------------------------------|---------------------|
| `RX`, `RY`, `RZ`     | Single qubit rotations. Notice that the interaction is on and this affects the resulting gate fidelity.                                                                        | rotation angle      |
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
parameter values allows to sample from the pulse sequence result.

```python exec="on" source="material-block" session="pulser-basic"
import torch
import matplotlib.pyplot as plt
from qadence import Register, QuantumCircuit, QuantumModel

register = Register(2)
circuit = QuantumCircuit(register, bell_state)
model = QuantumModel(circuit, backend="pulser", diff_mode="gpsr")

params = {
    "wait": torch.tensor([383]),  # ns
    "y": torch.tensor([torch.pi/2]),
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

At variance with other backends, the Pulser one provides the concept of `Device`, borrowed from the [`pulser`](https://pulser.readthedocs.io/en/stable/) library.

A `Device` instance encapsulate all the properties defining a real neutral atoms processor, including but not limited to the maximum laser amplitude for the pulses, the maximum distance between two qubits and the maximum duration of the pulse.

`qadence` offers a simplified interface with only two devices which can be found [here][qadence.backends.pulser.devices]

* `REALISTIC` (default): device specification very similar to a real neutral atom quantum processor.
* `IDEALIZED`: ideal device which should be used only for testing purposes. It does not have any limitation in what can be run with it.

One can use the `Configuration` of the Pulser backend to select the appropriate device:

```python exec="on" source="material-block" session="pulser-basic"
from qadence.backends.pulser.devices import Device

register = Register(2)
circuit = QuantumCircuit(register, bell_state)

model = QuantumModel(
    circuit,
    backend="pulser",
    diff_mode="gpsr",
    configuration={"device_type": Device.IDEALIZED}
)

# alternatively directly one of the devices available in Pulser
# can also be supplied in the same way
from pulser.devices import AnalogDevice

model = QuantumModel(
    circuit,
    backend="pulser",
    diff_mode="gpsr",
    configuration={"device_type": AnalogDevice}
)
```

## Create your own gate
A big advantage of the `chain` block is it makes it easy to create complex
operations from simple ones. Take the entanglement operation as an example.

The operation consists of moving _all_ the qubits to the `X` basis having the
atoms' interaction perform a controlled-Z operation during the free evolution.
And we can easily recreate this pattern using the `wait` (corresponding to free
evolution) and `AnalogRY` blocks with appropriate parameters.

```python exec="on" source="material-block" session="pulser-basic"
from qadence import AnalogRY, chain, wait

def my_entanglement(duration):
    return chain(
        AnalogRY(-torch.pi / 2),
        wait(duration)
    )
```

Then we proceed as before.

```python exec="on" source="material-block" session="pulser-basic" html="1"
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

```python exec="on" source="material-block" html="1" session="pulser-basic"
model.assign_parameters(params).draw(draw_phase_area=True, show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


## Large qubits registers
The constructor `Register(n_qubits)` generates a linear register that works fine
with two or three qubits. But for the blocks we have so far, large registers
work better with a square loop layout like the following.

```python exec="on" source="material-block" html="1" session="pulser-basic"
register = Register.square(qubits_side=4)
register.draw(show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

In those cases, global pulses are preferred to generate entanglement to avoid
changing the addressing pattern on the fly.

```python exec="on" source="material-block" html="1" session="pulser-basic"
protocol = chain(
    entangle("t"),
    AnalogRY(torch.pi / 2),
)

register = Register.square(qubits_side=2)
circuit = QuantumCircuit(register, protocol)
model = QuantumModel(circuit, backend="pulser", diff_mode="gpsr")
model.backend.backend.config.with_modulation = True

params = {
    "t": torch.tensor([1956]),  # ns
}

sample = model.sample(params, n_shots=500)[0]

fig, ax = plt.subplots()
ax.bar(sample.keys(), sample.values())
plt.xticks(rotation='vertical')
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```
```python exec="on" source="material-block" html="1" session="pulser-basic"
model.assign_parameters(params).draw(draw_phase_area=True, show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

!!! note
    The gates shown here don't work with arbitrary registers since they rely on
    the registered geometry to work properly.

## Working with observables

TODO

## Hardware efficient ansatz
<!-- The current backend version does not support Qadence `Observables`. However, it's
still possible to use the regular Pulser simulations to calculate expected
values.

To do so, we use the `assign_parameters` property to recover the Pulser sequence.

```python exec="on" source="material-block" session="pulser-basic"
params = {
    "t": torch.tensor([383]),  # ns
}

built_sequence = model.assign_parameters(params)
```


Next, we create the desired observables using `Qutip` [^2].


```python exec="on" source="material-block" session="pulser-basic"
import qutip

zz = qutip.tensor([qutip.qeye(2), qutip.sigmaz(), qutip.qeye(2), qutip.sigmaz()])
xy = qutip.tensor([qutip.qeye(2), qutip.sigmax(), qutip.qeye(2), qutip.sigmay()])
yx = qutip.tensor([qutip.qeye(2), qutip.sigmay(), qutip.qeye(2), qutip.sigmax()])
```


We use the Pulser `Simulation` class to run the sequence and call the method `expect` over our observables.

```python exec="on" source="material-block" result="json" session="pulser-basic"
from pulser_simulation import Simulation

sim = Simulation(built_sequence)
result = sim.run()

final_result = result.expect([zz, xy + yx])
print(final_result[0][-1], final_result[1][-1])
```

We use the Pulser `Simulation` class to run the sequence and call the method
`expect` over our observables.

Here the `final_result` contains the expected values during the evolution (one
point per nanosecond). In this case, looking only at the final values, we see
the qubits `q0` and `q2` are on the *Bell-diagonal* state $\Big(|00\rangle -i |11\rangle\Big)/\sqrt{2}$. -->

Finally, let's put all together by constructing a digital-analog version of a hardware
efficient ansatz.
```python exec="on" source="material-block" html="1" session="pulser-basic"
from qadence import fourier_feature_map, RX, RY

protocol = chain(
    fourier_feature_map(1, param="x"),
    entangle("t", qubit_support=(0,1)),
    RY(0, "th1"),
    RX(0, "th2"),
)

register = Register(2)
circuit = QuantumCircuit(register, protocol)
model = QuantumModel(circuit, backend="pulser", diff_mode='gpsr')

params = {
    "x": torch.tensor([0.8]),
    "t": torch.tensor([900]), # ns
    "th1":  torch.tensor([1.5]),
    "th2":  torch.tensor([0.9])
}

model.assign_parameters(params).draw(draw_phase_area=True, show=False)
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## References

[^1]: [Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays](https://pulser.readthedocs.io/en/stable/)
[^2]: [Qutip](https://qutip.org)
