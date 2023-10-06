**Qadence** is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable qubit interaction defined on _**arbitrary register topologies**_ realizable on neutral atom devices.

[![pre-commit](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/lint.yml)
[![tests](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml/badge.svg)](https://github.com/pasqal-io/qadence/actions/workflows/test_fast.yml)

## Feature highlights

* A [block-based system](tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and scalable manner, inspired by the Julia quantum SDK
  [Yao.jl](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](digital_analog_qc/analog-basics.md) to work with _**interacting qubit systems**_
  using [arbitrary registers topologies](tutorials/register.md).

* An intuitive [expression-based system](tutorials/parameters.md) developed on top of the symbolic library [Sympy](https://www.sympy.org/en/index.html) to construct _**parametric quantum programs**_ easily.

* [High-order generalized parameter shift rules](link to psr tutorial) for _**differentiating parametrized quantum operations**_.

* Out-of-the-box _**automatic differentiability**_ of quantum programs with [PyTorch](https://pytorch.org/) integration.

* _**Efficient execution**_ on a variety of different purpose backends: from state vector simulators to tensor network emulators and real devices.


## Citation

If you use Qadence for a publication, we kindly ask you to cite our work using the bibtex citation:

```
@misc{qadence2023pasqal,
  url = {https://github.com/pasqal-io/qadence},
  title = {Qadence: {A} {D}igital-analog quantum programming interface.},
  year = {2023}
}
```

In following are some rudimentary examples of Qadence possibilites in the digital, analog and digital-analog paradigms.

## Sampling the canonical Bell state

This example illustrates how to prepare a [Bell state](https://en.wikipedia.org/wiki/Bell_state) using digital gates and sampling from the outcome bitstring distribution:

```python exec="on" source="material-block" result="json"
import torch # markdown-exec: hide
torch.manual_seed(0) # markdown-exec: hide
from qadence import CNOT, H, chain, sample

# Preparing a Bell state by composing a Hadamard and CNOT gates in sequence.
bell_state = chain(H(0), CNOT(0,1))

# Sample with 100 shots.
samples = sample(bell_state, n_shots=100)
print(f"samples = {samples}") # markdown-exec: hide
from qadence.divergences import js_divergence # markdown-exec: hide
from collections import Counter # markdown-exec: hide
js = js_divergence(samples[0], Counter({"00":50, "11":50})) # markdown-exec: hide
assert js < 0.005 # markdown-exec: hide
```

## Analog emulation of a perfect state transfer

This next example showcases the construction and sampling of a system that admits a perfect state transfer between the two edge qubits of a three qubit register laid out in a
line. This relies on time-evolving a Hamiltonian for a custom defined qubit interation until $t=\frac{\pi}{\sqrt 2}$.

```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import X, Y, HamEvo, Register, product_state, sample, add

# Define the qubit-qubit interaction term.
def interaction(i, j):
    return 0.5 * (X(i) @ X(j) + Y(i) @ Y(j))  # Compose gates in parallel and sum their contribution.

# Initial state with left-most qubit in the 1 state.
init_state = product_state("100")

# Define a register of 3 qubits laid out in a line.
register = Register.line(n_qubits=3)

# Define an interaction Hamiltonian by summing interactions on indexed qubits.
# hamiltonian = interaction(0, 1) + interaction(1, 2)
hamiltonian = add(interaction(*edge) for edge in register.edges)

# Define and time-evolve the Hamiltonian until t=pi/sqrt(2).
t = pi/(2**0.5)  # Dimensionless.
evolution = HamEvo(hamiltonian, t)

# Sample with 100 shots.
samples = sample(register, evolution, state=init_state, n_shots=100)
print(f"{samples = }") # markdown-exec: hide
from collections import Counter # markdown-exec: hide
assert samples[0] == Counter({"001": 100}) # markdown-exec: hide
```

## Digital-analog example

This final example deals with the construction and sampling of an Ising Hamiltonian that includes a distance-based interaction between qubits and a global analog block of rotations around the X-axis. Here, global has to be understood as applied to the whole register. <Specify the distance unit.>

```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import Register, AnalogRX, sample

# Global analog RX block.
block = AnalogRX(pi)

# Almost non-interacting qubits as too far apart.
register = Register.from_coordinates([(0,0), (0,15)])  # Dimensionless.
samples = sample(register, block)
print(f"distance = 15: {samples = }") # markdown-exec: hide
from collections import Counter # markdown-exec: hide
from qadence.divergences import js_divergence # markdown-exec: hide
js = js_divergence(samples[0], Counter({"11": 100})) # markdown-exec: hide
assert js < 0.01 # markdown-exec: hide

# Interacting qubits as close together.
register = Register.from_coordinates([(0,0), (0,5)])
samples = sample(register, AnalogRX(pi))
print(f"distance =  5: {samples = }") # markdown-exec: hide
js = js_divergence(samples[0], Counter({"01":33, "10":33, "00":33, "11":1})) # markdown-exec: hide
assert js < 0.05 # markdown-exec: hide```
```

## Further Resources

For a more comprehensive introduction and advanced topics, please have a look at the following tutorials:

* [Quantum state conventions](tutorials/state_conventions.md) used throughout **Qadence**.
* [Basic tutorials](tutorials/getting_started.md) for first hands-on.
* [Digital-analog basics](digital_analog_qc/analog-basics.md) to build quantum programs in the digital-analog paradigm.
* [Parametric quantum circuits](tutorials/parameters.md) for the generation and manipulation of parametric programs.
* [Advanced features](advanced_tutorials) about low-level backend interface and differentiablity.
* [`QuantumModel`](advanced_tutorials/custom-models.md) for defining custom models.

## Installation guide

Qadence can be install with `pip` from PyPI as follows:

```bash
pip install qadence
```

The default backend for Qadence is [PyQTorch](https://github.com/pasqal-io/pyqtorch), a differentiable state vector simulator for digital-analog simulation. It is possible to install additional backends and the circuit visualization library using the following extras:

* `braket`: the [Braket](https://github.com/amazon-braket/amazon-braket-sdk-python) backend.
* `pulser`: the [Pulser](https://github.com/pasqal-io/Pulser) backend for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: to display diagrammatically quantum circuits.

by running:

```bash
pip install qadence[braket, pulser, visualization]
```

!!! warning
    In order to correctly install the `visualization` extra, the `graphviz` package needs to be installed
    in your system:

    ```bash
    # on Ubuntu
    sudo apt install graphviz

    # on MacOS
    brew install graphviz

    # via conda
    conda install python-graphviz
    ```
---
