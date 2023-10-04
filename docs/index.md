<h1 style="display: none;">noheading</h1>

!!! warning "Large Logo"
    Put a large version of the logo here.

**Qadence** is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable interaction defined on _**arbitrary qubit register topologies**_ realisable on neutral atom devices.

## Feature highlights

* A [block-based system](tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and extensible manner. Heavily inspired by
  [**Yao.jl**](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](digital_analog_qc/analog-basics.md) to work with _**interacting qubit systems**_
  using [arbitrary qubit registers topologies](tutorials/register.md).

* Intuitive, [expression-based system](tutorials/parameters.md) built on top of [**Sympy**](https://www.sympy.org/en/index.html) to construct
  _**parametric quantum programs**_.

* [Higher-order generalized parameter shift](link to psr tutorial) rules for _**differentiating
  arbitrary parametrized quantum operations**_ on real hardware.

* Out-of-the-box automatic differentiability of quantum programs with [**PyTorch**](https://pytorch.org/) integration. **PyTorch** also constitues our numerical backend throughout.

* The `QuantumModel` abstraction to make `QuantumCircuit` differentiable and runnable on a variety of different purpose
  backends: state vector simulators, tensor network emulators and real devices.


## In a nutshell
Documentation can be found here: [https://pasqal-qadence.readthedocs-hosted.com/en/latest](https://pasqal-qadence.readthedocs-hosted.com/en/latest).

## Remarks
Quadence uses `torch.float64` as the default datatype for tensors (`torch.complex128` for complex tensors).

Here are some examples of **Qadence** possibilites in both the digital and digital-analog paradigms.

### Bell state

Sample from the canonical [Bell state](https://en.wikipedia.org/wiki/Bell_state).

```python exec="on" source="material-block" result="json"
import torch # markdown-exec: hide
torch.manual_seed(0) # markdown-exec: hide
from qadence import CNOT, H, chain, sample

samples = sample(chain(H(0), CNOT(0,1)), n_shots=100)
print(samples) # markdown-exec: hide
from qadence.divergences import js_divergence # markdown-exec: hide
from collections import Counter # markdown-exec: hide
js = js_divergence(samples[0], Counter({"00":50, "11":50})) # markdown-exec: hide
assert js < 0.005 # markdown-exec: hide
```

### Digital-analog emulation

#### Perfect state transfer

Construct and sample a system that admits perfect state transfer between the two edge qubits of a register laid out in a
line, by solving Hamiltonian evolution for custom qubit interation until time $t=\frac{\pi}{\sqrt 2}$.

```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import X, Y, HamEvo, Register, product_state, sample, add

# Define qubit-qubit interaction term.
def interaction(i, j):
    return 0.5 * (X(i) @ X(j) + Y(i) @ Y(j))

# Initial state with left-most qubit in the 1 state.
init_state = product_state("100")

# Register of qubits laid out in a line.
register = Register.line(n_qubits=3)

# Interaction Hamiltonian. Identical to:
# hamiltonian = interaction(0, 1) + interaction(1, 2)
hamiltonian = add(interaction(*edge) for edge in register.edges)

# Define a time-dependent Hamiltonian and evolve until t=pi/sqrt(2).
t = pi/(2**0.5)
evolution = HamEvo(hamiltonian, t)

samples = sample(register, evolution, state=init_state, n_shots=1)
print(f"{samples = }") # markdown-exec: hide
from collections import Counter # markdown-exec: hide
assert samples[0] == Counter({"001": 1}) # markdown-exec: hide
```

#### <Nice example title>

Construct and sample an Ising Hamiltonian that includes a distance-based interaction between qubits and a global analog block of RX. Here, global has to be understood as applied to the whole register. <Specify the distance unit.>

```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import Register, AnalogRX, sample

# Global analog RX block.
block = AnalogRX(pi)

# Almost non-interacting qubits as too far apart.
register = Register.from_coordinates([(0,0), (0,15)])
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

Qadence can be install with `pip` as follows:

```bash
export TOKEN_USERNAME=MYUSERNAME
export TOKEN_PASSWORD=THEPASSWORD

pip install --extra-index-url "https://${TOKEN_USERNAME}:${TOKEN_PASSWORD}@gitlab.pasqal.com/api/v4/projects/190/packages/pypi/simple" qadence[pulser,visualization]
```

where the token username and password can be generated on the
[Gitlab UI](https://gitlab.pasqal.com/-/profile/personal_access_tokens). Remember to give registry read/write permissions to the generated token.

The default backend for qadence is pyqtorch (a differentiable state vector simulator).
You can install one or all of the following additional backends and the circuit visualization library using the following extras:

* `braket`: install the Amazon Braket quantum backend
* `pulser`: install the Pulser backend. Pulser is a framework for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: install the library necessary to visualize quantum circuits.

!!! warning
    In order to correctly install the "visualization" extra, you need to have `graphviz` installed
    in your system. This depends on the operating system you are using:

    ```bash
    # on Ubuntu
    sudo apt install graphviz

    # on MacOS
    brew install graphviz

    # via conda
    conda install python-graphviz
    ```
---
