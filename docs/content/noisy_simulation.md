# Noisy Simulation

Running programs on NISQ devices often leads to partially useful results due to the presence of noise. In order to perform realistic simulations, a number of noise models (for digital operations, analog operations and simulated readout errors) are supported in Qadence.

Noisy simulations shift the quantum paradigm from a close-system to an open-system represented by a probabilistic combination $p_i$ of possible pure states $|\psi_i \rangle$. Thus, the system is described by a density matrix $\rho$ defined as follows:

$$
\rho = \sum_i p_i |\psi_i\rangle \langle \psi_i|
$$

The transformations of the density operator of an open quantum system interacting with its environment (noise) are represented by the super-operator $S: \rho \rightarrow S(\rho)$. Any arbitrary super-operator can be decomposed in the following form:

$$
S(\rho) = \sum_i K_i \rho K^{\dagger}_i
$$

Where $K_i$ are the Kraus operators, and satisfy the property $\sum_i K_i K^{\dagger}_i = \mathbb{I}$. As noise is the result of system interactions with its environment, it is therefore possible to simulate noisy quantum circuit with noisy type of gates.

## Noise Protocols

The noise protocols applicable in `Qadence` are classified into three types: digital, analog, and readout error. Digital noise in quantum computers occurs due to unintended state changes during the gate operations such as `BitFlip` or `AmplitudeDamping`. Analog noise is when qubits lose coherence due to environmental interactions, as in `Depolarizing`. Readout error arises during the measurement process, leading to incorrect state assignment due to imperfections in the detection mechanism.

When dealing with programs involving digital operations(including readout errors), `Qadence` has interface to noise models implemented in `PyQTorch`. The following is a list of the supported types of noise. Detailed equation of noise options are available from [PyQTorch](https://pasqal-io.github.io/pyqtorch/latest/noise/). In terms of analog noise, both `PyQTorch` and `Pulser` backends are supporting the noisy simulation operation. For `Pulser` noise implementation, you can refer to [Pulser](https://pulser.readthedocs.io/en/stable/tutorials/noisy_sim.html).

Each noise protocol requires specific `options` parameter for its usage which can be found [here](https://github.com/pasqal-io/qadence/blob/main/qadence/noise/protocols.py).

### Digital noise protocol

The following are the protocols of supported digital noise, along with brief descriptions. For digital noise, the `error_probability` is necessary for the noise initialization at the `options` parameter.

- BITFLIP: flips between |0⟩ and |1⟩ with `error_probability`

- PHASEFLIP: flips the phase of a qubit by applying a Z gate with `error_probability`

- DEPOLARIZING: randomizes the state of a qubit by applying I, X, Y, or Z gates with equal `error_probability`

- PAULI_CHANNEL: applies one of the Pauli operators (X, Y, Z) to a qubit with specified `error_probabilities`

- AMPLITUDE_DAMPING: transitions qubit status from |1⟩ to |0⟩ with `error_probability`

- PHASE_DAMPING: reduces coherence by randomizing the phase with `error_probability`

- GENERALIZED_AMPLITUDE_DAMPING: extends amplitude damping; the first float is `error_probability` of amplitude damping, and second float is the `damping_rate`

For digital noise simulation, you need to state `NoiseProtocol` with `DIGITAL` and then specify the noise type. Also, you put the value of `error_probability` as in next example.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseProtocol

protocol = NoiseProtocol.DIGITAL.DEPOLARIZING
options = {"error_probability": 0.1}
```

### Analog noise protocol

Qadence is in the process of fully supporting all the noise protocols in the backend(especially `Pulser`). However, we are in transition, and currently, only DEPOLARIZING is available for analog noise protocol. The `noise_probs` is necessary for the noise initialization at the `options` parameter.

- Depolarizing: evolves to the maximally mixed state with `noise_probs`

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseProtocol

protocol = NoiseProtocol.ANALOG.DEPOLARIZING
options = {"noise_probs": 0.1}
```

### Readout error protocol

Readout errors are linked to the incorrect identification of the final state of the qubits. This is computed with the density matrix of the state through `sample` execution. We have a `seed` parameter in `options` for reproducible purposes.

- Independent: all bits are corrupted with an equal `error_probability`
- Correlated: apply `confusion_matrix` on error probabilities

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseProtocol

protocol=NoiseProtocol.READOUT.INDEPENDENT
options = {"error_probability": 0.01, "seed": 0}
```

## NoiseHandler

In order to apply noise protocols to gates or circuits, we need a wrapper called the `NoiseHandler` type. It is a container of several noise instances that require a specific `protocol` and a dictionary of `options` (or lists). The `protocol` field is to be instantiated from `NoiseProtocol` and `options` includes error-related information such as `error_probability`, `noise_probs`, and `seed`.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseHandler, NoiseProtocol

digital_noise = NoiseHandler(protocol=NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, options={"error_probability": 0.1})
analog_noise = NoiseHandler(protocol=NoiseProtocol.ANALOG.DEPOLARIZING, options={"noise_probs": 0.1})
readout_noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": 0.1, "seed": 0})
```

### Using NoiseHandlers in batch

`NoiseHandler` can be used in a more compact way to represent noise in batches.

- A `NoiseHandler` can be initiated with a list of protocols and a list of options (careful with the order)
- A `NoiseHandler` can be appended to other `NoiseHandler` instances

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseHandler, NoiseProtocol

# initiating with list of protocols and options
protocols = [NoiseProtocol.DIGITAL.DEPOLARIZING, NoiseProtocol.READOUT]
options = [{"error_probability": 0.1}, {"error_probability": 0.1, "seed": 0}]

noise_handler_list = NoiseHandler(protocols, options)
print(noise_handler_list)  # markdown-exec: hide

# NoiseHandler appending
depo_noise = NoiseHandler(protocol=NoiseProtocol.DIGITAL.DEPOLARIZING, options={"error_probability": 0.1})
readout_noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": 0.1, "seed": 0})

noise_combination = NoiseHandler(protocol=NoiseProtocol.DIGITAL.BITFLIP, options={"error_probability": 0.1})
noise_combination.append([depo_noise, readout_noise])
print(noise_combination)  # markdown-exec: hide
```

!!! warning "NoiseHandler scope"
    Note it is not possible to define `NoiseHandler` instances with both digital and analog noises, both readout and analog noises, several analog noises, several readout noises, or a readout noise that is not the last defined protocol within `NoiseHandler`.


## Executing Noisy Simulation

Noise simulation can be structured by applying a `NoiseHandler` to the desired `gate`, `block`, `circuit`, or `model`.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseProtocol, RX, run, NoiseHandler
import torch

noise = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.2})
op = RX(0, torch.pi, noise = noise)

print(f"Noisy density matrix = {run(op)}")  # markdown-exec: hide
```

We can also apply noise with `set_noise` function that apply given noise to the whole object.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import set_noise, kron, QuantumCircuit, RX, NoiseHandler, NoiseProtocol, run

n_qubits = 2
block = kron(RX(i, f"theta_{i}") for i in range(n_qubits))
circuit = QuantumCircuit(2, block)
noise = NoiseHandler(NoiseProtocol.DIGITAL.PHASEFLIP, {"error_probability": 0.1})

# The function changes the circuit in place:
set_noise(circuit, noise)

print(f"Noisy density matrix = {run(circuit)}") # markdown-exec: hide
```

let's say we want to apply noise only to specific type of gates, a `target_class` argument can be passed with the corresponding block in `set_noise`.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import X, chain, set_noise, NoiseHandler, NoiseProtocol

block = chain(RX(0, "theta"), X(0))
noise = NoiseHandler(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, {"error_probability": 0.1})
set_noise(block, noise, target_class=X)

for block in block.blocks: # markdown-exec: hide
    print(f"Noise type for block {block} is {block.noise}.") # markdown-exec: hide
```

One can set different noise models for each individual gates within the same circuit as follows:

```python exec="on" source="material-block" session="noise" result="json" html="1"
from qadence import QuantumCircuit, X, sample, kron, NoiseHandler, NoiseProtocol
import matplotlib.pyplot as plt

n_qubits = 2
noise_bitflip = NoiseHandler(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.1})
noise_amplitude_damping = NoiseHandler(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, {"error_probability": 0.3})
block = kron(X(0, noise=noise_bitflip), X(1, noise=noise_amplitude_damping))
circuit = QuantumCircuit(n_qubits, block)

n_shots=1000
xs = sample(circuit, n_shots=n_shots)

items = list(xs[0].keys())
values = [v/n_shots for v in xs[0].values()]

plt.figure()
plt.bar(range(len(values)), values, color='blue', alpha=0.7)
plt.xticks(range(len(items)), items)
plt.title("Probability of state occurrence")
plt.xlabel('Possible States')
plt.ylabel('Probability')
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

The result of this figure would be a 100% `11` state without noise. However, with `X(0)` bitflip noise, the state `01` has some possibility, and with `X(1)` amplitude damping noise, more gap appears between state pairs of (`00`, `01`) and (`10`, `11`), as shown in the figure.


### Digital noisy simulation

To use the digital noise option, one needs to state the `protocol` and `error_probability` in `NoiseHandler`. All noise types require at least a single float for the error_probability.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import H, QuantumCircuit, QuantumModel, NoiseProtocol, NoiseHandler, PI, BackendName, expectation

digital_block = H(0)
circuit = QuantumCircuit(1, digital_block)

options = {"error_probability": 0.3}
noise = NoiseHandler(protocol=NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, options=options)

model_digital_noisy = QuantumModel(
    circuit=circuit,
    backend=BackendName.PYQTORCH,
    noise=noise,
)

noisy_sample = model_digital_noisy.sample()
print(f"noisy_sample = {noisy_sample}") # markdown-exec: hide
```

Without the noise, sample results should have a near 50% probability of both states. In the presence of AMPLITUDE_DAMPING noise, state 0 has a higher probability than state 1.

### Analog noisy simulation

Analog noise simulations must declare a `protocol` and a `noise_probs` at the `NoiseHandler` level.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import DiffMode, NoiseHandler, QuantumModel
from qadence.blocks import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.operations import AnalogRX, AnalogRZ, Z
from qadence.types import PI, BackendName, NoiseProtocol

analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
observable = Z(0) + Z(1)
circuit = QuantumCircuit(2, analog_block)

options = {"noise_probs": 0.1}
noise = NoiseHandler(protocol=NoiseProtocol.ANALOG.DEPOLARIZING, options=options)
model_noisy = QuantumModel(
    circuit=circuit,
    observable=observable,
    backend=BackendName.PULSER,
    diff_mode=DiffMode.GPSR,
    noise=noise,
)
noisy_expectation = model_noisy.expectation()
print(f"noisy_expectation = {noisy_expectation}") # markdown-exec: hide
```

At the moment, analog noisy simulations are only compatible with the Pulser backend, but we are in the process of supporting the PyQTorch backend with more noise protocols.

### Readout error simulation

State Preparation and Measurement (SPAM) in the hardware is a major source of noise in the execution of quantum programs. They are typically described using confusion matrices of the form:

$$
T(x|x')=\delta_{xx'}
$$

Qadence offers to simulate readout errors with the `NoiseHandler` to corrupt the output
samples of a simulation, through execution via a `QuantumModel`:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import QuantumModel, QuantumCircuit, kron, H, Z
from qadence import hamiltonian_factory

# Simple circuit and observable construction.
block = kron(H(0), Z(1))
circuit = QuantumCircuit(2, block)
observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

# Construct a quantum model.
model = QuantumModel(circuit=circuit, observable=observable)

# Define a noise model to use.
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT)

# Run noiseless and noisy simulations.
noiseless_samples = model.sample(n_shots=100)
noisy_samples = model.sample(noise=noise, n_shots=100)

print(f"noiseless = {noiseless_samples}") # markdown-exec: hide
print(f"noisy = {noisy_samples}") # markdown-exec: hide
```

It is possible to pass options to the noise model. In the previous example, a noise matrix is implicitly computed from a uniform distribution.

For `NoiseProtocol.READOUT.INDEPENDENT`, the `option` dictionary argument accepts the following options:

- `seed`: defaulted to `None`, for reproducibility purposes
- `error_probability`: If float, the same probability is applied to every bit. By default, this is 0.1.
    If a 1D tensor with the number of elements equal to the number of qubits, a different probability can be set for each qubit. If a tensor of shape (n_qubits, 2, 2) is passed, that is a confusion matrix obtained from experiments, we extract the error_probability.
    and do not compute internally the confusion matrix as in the other cases.
- `noise_distribution`: defaulted to `WhiteNoise.UNIFORM`, for non-uniform noise distributions

For `NoiseProtocol.READOUT.CORRELATED`, the `option` dictionary argument accepts the following options:
- `confusion_matrix`: The square matrix representing $T(x|x')$ for each possible bitstring of length `n` qubits. Should be of size ($2^n, 2^n$).
- `seed`: defaulted to `None`, for reproducibility purposes


Noisy simulations go hand-in-hand with measurement protocols. In this case, both measurement and noise protocols have to be defined appropriately. Please note that a measurement protocol without a noise protocol will be ignored for expectation values computations.


```python exec="on" source="material-block" session="noise" result="json"
from qadence.measurements import Measurements

# Define a noise model with options.
options = {"error_probability": 0.01}
noise = NoiseHandler(protocol=NoiseProtocol.READOUT.INDEPENDENT, options=options)

# Define a tomographical measurement protocol with options.
options = {"n_shots": 10000}
measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options=options)

# Run noiseless and noisy simulations.
noiseless_exp = model.expectation(measurement=measurement)
noisy_exp = model.expectation(measurement=measurement, noise=noise)

print(f"noiseless = {noiseless_exp}") # markdown-exec: hide
print(f"noisy = {noisy_exp}") # markdown-exec: hide
```
