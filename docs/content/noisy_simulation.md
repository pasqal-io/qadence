# Noisy Simulation

Running programs on NISQ devices often leads to imperfect results due to the presence of noise. In order to perform realistic simulations, a number of noise models (for digital operations, analog operations and simulated readout errors) are supported in `Qadence`.

Noisy simulations shift the quantum paradigm from a close-system (noiseless case) to an open-system (noisy case) where a quantum system is represented by a probabilistic combination $p_i$ of possible pure states $|\psi_i \rangle$. Thus, the system is described by a density matrix $\rho$ (and computation modify the density matrix) defined as follows:

$$
\rho = \sum_i p_i |\psi_i\rangle \langle \psi_i|
$$

The noise protocols applicable in `Qadence` are classified into three types: digital (for digital operations), analog (for analog operations), and readout error (for measurements).

## Specifying a noise protocol

Each noise protocol can be specified using `NoiseProtocol` and requires specific `options` parameter passed as a dictionary. We show below for each type of noise how this can be done.

### Digital noise protocol

Digital noise refer to unintended changes occurring with reference to the application of a noiseless digital gate operation. The following are the protocols of supported digital noise, along with brief descriptions. For digital noise, the `error_probability` is necessary for the noise initialization at the `options` parameter.

When dealing with programs involving digital operations, `Qadence` has interface to noise models implemented in `PyQTorch`.
Detailed equations for these protocols are available from [PyQTorch](https://pasqal-io.github.io/pyqtorch/latest/noise/).

- BITFLIP: flips between |0⟩ and |1⟩ with `error_probability`
- PHASEFLIP: flips the phase of a qubit by applying a Z gate with `error_probability`
- DEPOLARIZING: randomizes the state of a qubit by applying I, X, Y, or Z gates with equal `error_probability`
- PAULI_CHANNEL: applies the Pauli operators (X, Y, Z) to a qubit with specified `error_probabilities`
- AMPLITUDE_DAMPING: models the asymmetric process through which the qubit state |1⟩ irreversibly decays into the state |0⟩ with `error_probability`
- PHASE_DAMPING: similar to AMPLITUDE_DAMPING but concerning the phase
- GENERALIZED_AMPLITUDE_DAMPING: extends amplitude damping; the first float is `error_probability` of amplitude damping, and second float is the `damping_rate`

For digital noise simulation, you need to state `NoiseProtocol` with `DIGITAL` and then specify the noise protocol. Also, you put the value of `error_probability` as in next example.

```python exec="on" source="material-block" session="output"
from qadence import NoiseProtocol

protocol = NoiseProtocol.DIGITAL.DEPOLARIZING
options = {"error_probability": 0.1}
```

### Analog noise protocol

Analog noise can be set for analog operations. At the moment, we only enabled simulations via the `Pulser` backend.
For `Pulser` noise implementation, you can refer to [Pulser](https://pulser.readthedocs.io/en/stable/tutorials/noisy_sim.html).
`Qadence` is in the process of fully supporting all the noise protocols in the backends (especially `Pulser`). However, we are in transition, and currently, only DEPOLARIZING and DEPHAZING are available as protocols. The `options` dictionary requires to specify the field `noise_probs`.

- Depolarizing: evolves to the maximally mixed state with `noise_probs`
- Dephasing: induces the loss of phase coherence without affecting the population of computational basis states

```python exec="on" source="material-block" session="output"
from qadence import NoiseProtocol

protocol = NoiseProtocol.ANALOG.DEPOLARIZING
options = {"noise_probs": 0.1}
```

### Readout error protocol

Readout errors are linked to the incorrect measurement outcomes from the system. In this protocol, we have `error_probability`, `confusion_matrix`, and `seed` option parameters. For the `error_probability` parameter, if float, the same probability error is applied to every bit. A different probability can be set for each qubit if a 1D tensor has an element number equal to the number of qubits. For `confusion_matrix` parameter, the square matrix for each possible bitstring of length `n` qubits. We have a `seed` parameter for reproducible purposes.

Currently, two readout protocols are available via [PyQTorch](https://pasqal-io.github.io/pyqtorch/latest/noise/).

- Independent: all bits are corrupted independently with each other.
- Correlated: apply a `confusion_matrix` of corruption between each possible bitstrings

```python exec="on" source="material-block" session="output"
from qadence import NoiseProtocol

protocol=NoiseProtocol.READOUT.INDEPENDENT
options = {"error_probability": 0.01, "seed": 0}
```

### Preparing noise protocols for usage

In order to apply the noise to `Qadence` objects, we need a wrapper called the `AbstractNoise` type. It is a container of several noise instances that require a specific `protocol` and a dictionary of `options` (or lists). The `protocol` field is to be instantiated from `NoiseProtocol` and `options` includes error-related information such as `error_probability`, `noise_probs`, and `seed`.

```python exec="on" source="material-block" session="output"
from qadence import AbstractNoise, NoiseProtocol

digital_noise = AbstractNoise(protocol=NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, options={"error_probability": 0.1})
analog_noise = AbstractNoise(protocol=NoiseProtocol.ANALOG.DEPOLARIZING, options={"noise_probs": 0.1})
readout_noise = AbstractNoise(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": 0.1, "seed": 0})
```

`AbstractNoise` can be used in a more compact way to represent noise in batches.

- A `AbstractNoise` can be initiated with a list of protocols and a list of options (careful with the order)
- A `AbstractNoise` can be appended to other `AbstractNoise` instances

```python exec="on" source="material-block" session="noise" result="json"
from qadence import AbstractNoise, NoiseProtocol

# initiating with list of protocols and options
protocols = [NoiseProtocol.DIGITAL.DEPOLARIZING, NoiseProtocol.READOUT]
options = [{"error_probability": 0.1}, {"error_probability": 0.1, "seed": 0}]

noise_handler_list = AbstractNoise(protocols, options)

# AbstractNoise appending
depo_noise = AbstractNoise(protocol=NoiseProtocol.DIGITAL.DEPOLARIZING, options={"error_probability": 0.1})
readout_noise = AbstractNoise(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": 0.1, "seed": 0})
noise_combination = AbstractNoise(protocol=NoiseProtocol.DIGITAL.BITFLIP, options={"error_probability": 0.1})

# prints noise_combination
noise_combination.append([depo_noise, readout_noise])
print(noise_combination)  # markdown-exec: hide
```

!!! warning "AbstractNoise scope"
    Note it is not possible to define `AbstractNoise` instances with both digital and analog noises, both readout and analog noises, several analog noises, several readout noises, or a readout noise that is not the last defined protocol within `AbstractNoise`.


## Executing Noisy Simulation

Noisy simulation can be set by applying a `AbstractNoise` to the desired `gate`, `block`, `QuantumCircuit`, or `QuantumModel`.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import NoiseProtocol, RX, run, AbstractNoise
import torch

noise = AbstractNoise(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.2})
circuit = RX(0, torch.pi, noise = noise)

# prints density matrix
run(circuit)
print(f"Noisy density matrix = {run(circuit)}")  # markdown-exec: hide
```

We can also apply noise with the `set_noise` function that apply a given noise configuration to the whole object.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import DiffMode, AbstractNoise, QuantumModel
from qadence.blocks import chain, kron
from qadence.circuit import QuantumCircuit
from qadence.operations import AnalogRX, AnalogRZ, Z
from qadence.types import PI, BackendName, NoiseProtocol
from qadence import set_noise

analog_block = chain(AnalogRX(PI / 2.0), AnalogRZ(PI))
observable = Z(0) + Z(1)
circuit = QuantumCircuit(2, analog_block)

noise = AbstractNoise(protocol=NoiseProtocol.ANALOG.DEPOLARIZING, options={"noise_probs": 0.2})
model = QuantumModel(
    circuit=circuit,
    observable=observable,
    backend=BackendName.PULSER,
    diff_mode=DiffMode.GPSR,
)

noiseless_expectation = model.expectation()

noisy_model = set_noise(model, noise)
noisy_expectation = noisy_model.expectation()
print(f"Noiseless expectation = {noiseless_expectation}") # markdown-exec: hide
print(f"Noisy expectation = {noisy_expectation}") # markdown-exec: hide
```

Let's say we want to apply noise only to specific type of gates, a `target_class` argument can be passed with the corresponding block in `set_noise`.

```python exec="on" source="material-block" session="noise" result="json"
from qadence import X, chain, set_noise, AbstractNoise, NoiseProtocol

block = chain(RX(0, "theta"), X(0))
noise = AbstractNoise(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, {"error_probability": 0.1})

# prints noise configuration for each gate
set_noise(block, noise, target_class=X)

for block in block.blocks: # markdown-exec: hide
    print(f"Noise type for gate {block} is {block.noise}.") # markdown-exec: hide
```

One can set different noise models for each individual gates within the same circuit as follows:

```python exec="on" source="material-block" session="noise" result="json" html="1"
from qadence import QuantumCircuit, X, sample, kron, AbstractNoise, NoiseProtocol
import matplotlib.pyplot as plt

n_qubits = 2
noise_bitflip = AbstractNoise(NoiseProtocol.DIGITAL.BITFLIP, {"error_probability": 0.1})
noise_amplitude_damping = AbstractNoise(NoiseProtocol.DIGITAL.AMPLITUDE_DAMPING, {"error_probability": 0.3})
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


The readout error is computed with the density matrix of the state through `sample` execution.

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
noise = AbstractNoise(protocol=NoiseProtocol.READOUT.INDEPENDENT, options={"error_probability": 0.1})

# Run noiseless and noisy simulations.
noiseless_samples = model.sample(n_shots=100)
noisy_samples = model.sample(noise=noise, n_shots=100)

print(f"noiseless = {noiseless_samples}") # markdown-exec: hide
print(f"noisy = {noisy_samples}") # markdown-exec: hide
```
