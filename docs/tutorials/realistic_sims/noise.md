Running programs on NISQ devices often leads to partially useful results due to the presence of noise.
In order to perform realistic simulations, a number of noise models are supported in Qadence and
corresponding error mitigation techniques whenever possible.

## Readout errors

State Preparation and Measurement (SPAM) in the hardware is a major source of noise in the execution of
quantum programs. They are typically described using confusion matrices of the form:

$$
T(x|x')=\delta_{xx'}
$$


Qadence offers to simulate readout errors with the `PostProcessingNoise` protocol to corrupt the output
samples of a simulation, through execution via a `QuantumModel`:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import QuantumModel, QuantumCircuit, kron, H, Z
from qadence import hamiltonian_factory
from qadence.noise import PostProcessingNoise

# Simple circuit and observable construction.
block = kron(H(0), Z(1))
circuit = QuantumCircuit(2, block)
observable = hamiltonian_factory(circuit.n_qubits, detuning=Z)

# Construct a quantum model.
model = QuantumModel(circuit=circuit, observable=observable)

# Define a noise model to use.
noise = PostProcessingNoise(protocol=PostProcessingNoise.READOUT)

# Run noiseless and noisy simulations.
noiseless_samples = model.sample(n_shots=100)
noisy_samples = model.sample(noise=noise, n_shots=100)

print(f"noiseless = {noiseless_samples}") # markdown-exec: hide
print(f"noisy = {noisy_samples}") # markdown-exec: hide
```

It is possible to pass options to the noise model. In the previous example, a noise matrix is implicitly computed from a
uniform distribution. The `option` dictionary argument accepts the following options:

- `seed`: defaulted to `None`, for reproducibility purposes
- `error_probability`: defaulted to 0.1, a bit flip probability
- `noise_distribution`: defaulted to `WhiteNoise.UNIFORM`, for non-uniform noise distributions
- `noise_matrix`: defaulted to `None`, if the noise matrix is known from third-party experiments, _i.e._ hardware calibration.

Noisy simulations go hand-in-hand with measurement protocols discussed in the previous [section](measurements.md), to assess the impact of noise on expectation values. In this case, both measurement and noise protocols have to be defined appropriately. Please note that a noise protocol without a measurement protocol will be ignored for expectation values computations.


```python exec="on" source="material-block" session="noise" result="json"
from qadence.measurements import Measurements

# Define a noise model with options.
options = {"error_probability": 0.01}
noise = PostProcessingNoise(protocol=PostProcessingNoise.READOUT, options=options)

# Define a tomographical measurement protocol with options.
options = {"n_shots": 10000}
measurement = Measurements(protocol=Measurements.TOMOGRAPHY, options=options)

# Run noiseless and noisy simulations.
noiseless_exp = model.expectation(measurement=measurement)
noisy_exp = model.expectation(measurement=measurement, noise=noise)

print(f"noiseless = {noiseless_exp}") # markdown-exec: hide
print(f"noisy = {noisy_exp}") # markdown-exec: hide
```

## Digital noisy simulation

When dealing with programs involving only digital operations, several options are made available from [PyQTorch](https://pasqal-io.github.io/pyqtorch/latest/noise/) via the `BlockNoiseType`. One can define noisy digital operations with `BlockNoise`as follows:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import BlockNoiseType, BlockNoise, RX, run
import torch

noise = BlockNoise(BlockNoiseType.BITFLIP, error_probability = 0.2)
noise = BlockNoise.bitflip(error_probability = 0.2) # equivalent

op = RX(0, torch.pi, noise = noise)

print(run(op))
```

It is also possible to set a noise configuration to gates within a composite block or circuit as follows:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import set_noise, chain

n_qubits = 2

block = chain(RX(i, f"theta_{i}") for i in range(n_qubits))

noise = BlockNoise.bitflip(error_probability = 0.1)

# The function changes the block in place:
set_noise(block, noise)
print(run(block))
```

There is an extra optional argument to specify the type of block we want to apply noise to. E.g., let's say we want to apply noise only to `X` gates, a `target_class` argument can be passed with the corresponding block:

```python exec="on" source="material-block" session="noise" result="json"
from qadence import X
block = chain(RX(0, "theta"), X(0))
set_noise(block, noise, target_class = X)

for block in block.blocks:
    print(block.noise)
```
