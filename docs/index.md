<h1 style="display: none;">noheading</h1>

!!! warning "Large Logo"
    Put a large verion of the logo herec.

Qadence is a Python package that provides a simple interface to build _**digital-analog quantum
programs**_ with tunable interaction defined on _**arbitrary qubit register layouts**_.

## Feature highlights

* A [block-based system](tutorials/getting_started.md) for composing _**complex digital-analog
  programs**_ in a flexible and extensible manner. Heavily inspired by
  [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl) and functional programming concepts.

* A [simple interface](digital_analog_qc/analog-basics.md) to work with _**interacting qubit systems**_
  using [arbitrary qubit registers](tutorials/register.md).

* Intuitive, [expression-based system](tutorials/parameters.md) built on top of `sympy` to construct
  _**parametric quantum programs**_.

* [Higher-order generalized parameter shift](link to psr tutorial) rules for _**differentiating
  arbitrary quantum operations**_ on real hardware.

* Out-of-the-box automatic differentiability of quantum programs using [https://pytorch.org](https://pytorch.org)

* `QuantumModel`s to make `QuantumCircuit`s differentiable and runnable on a variety of different
  backends like state vector simulators, tensor network emulators and real devices.

Documentation can be found here: [https://pasqal-qadence.readthedocs-hosted.com/en/latest](https://pasqal-qadence.readthedocs-hosted.com/en/latest).

## Remarks
Quadence uses torch.float64 as the default datatype for tensors (torch.complex128 for complex tensors).

## Examples

### Bell state

Sample from the [Bell state](https://en.wikipedia.org/wiki/Bell_state) in one line.

```python exec="on" source="material-block" result="json"
import torch # markdown-exec: hide
torch.manual_seed(0) # markdown-exec: hide
from qadence import CNOT, H, chain, sample

xs = sample(chain(H(0), CNOT(0,1)), n_shots=100)
print(xs) # markdown-exec: hide
from qadence.divergences import js_divergence # markdown-exec: hide
from collections import Counter # markdown-exec: hide
js = js_divergence(xs[0], Counter({"00":50, "11":50})) # markdown-exec: hide
assert js < 0.005 # markdown-exec: hide
```


### Perfect state transfer

We can construct a system that admits perfect state transfer between the two edge qubits in a
line of qubits at time $t=\frac{\pi}{\sqrt 2}$.
```python exec="on" source="material-block" result="json"
import torch
from qadence import X, Y, HamEvo, Register, product_state, sample, add

def interaction(i, j):
    return 0.5 * (X(i) @ X(j) + Y(i) @ Y(j))

# initial state with left-most qubit in the 1 state
init_state = product_state("100")

# register with qubits in a line
reg = Register.line(n_qubits=3)

# a line hamiltonian
hamiltonian = add(interaction(*edge) for edge in reg.edges)
# which is the same as:
# hamiltonian = interaction(0, 1) + interaction(1, 2)

# define a hamiltonian evolution over t
t = torch.pi/(2**0.5)
evolution = HamEvo(hamiltonian, t)

samples = sample(reg, evolution, state=init_state, n_shots=1)
print(f"{samples = }") # markdown-exec: hide
from collections import Counter # markdown-exec: hide
assert samples[0] == Counter({"001": 1}) # markdown-exec: hide
```


### Digital-analog emulation

Just as easily we can simulate an Ising hamiltonian that includes an interaction term based on the
distance of two qubits.  To learn more about digital-analog quantum computing see the
[digital-analog section](/digital_analog_qc/analog-basics.md).
```python exec="on" source="material-block" result="json"
from torch import pi
from qadence import Register, AnalogRX, sample

# global, analog RX block
block = AnalogRX(pi)

# two qubits far apart (practically non-interacting)
reg = Register.from_coordinates([(0,0), (0,15)])
samples = sample(reg, block)
print(f"distance = 15: {samples = }") # markdown-exec: hide
from collections import Counter # markdown-exec: hide
from qadence.divergences import js_divergence # markdown-exec: hide
js = js_divergence(samples[0], Counter({"11": 100})) # markdown-exec: hide
assert js < 0.01 # markdown-exec: hide

# two qubits close together (interacting!)
reg = Register.from_coordinates([(0,0), (0,5)])
samples = sample(reg, AnalogRX(pi))
print(f"distance =  5: {samples = }") # markdown-exec: hide
js = js_divergence(samples[0], Counter({"01":33, "10":33, "00":33, "11":1})) # markdown-exec: hide
assert js < 0.05 # markdown-exec: hide```
```


## Further Resources
For a more comprehensive introduction and advanced topics, we suggest you to
look at the following tutorials:

* [Description of quantum state conventions.](tutorials/state_conventions.md)
* [Basic tutorial](tutorials/getting_started.md) with a lot of detailed information
* Building [digital-analog](digital_analog_qc/analog-basics.md) quantum programs with interacting qubits
* [The sharp bits](tutorials/parameters.md) of creating parametric programs and observables
* [Advanced features](advanced_tutorials) like the low-level backend interface and model extremization
* Building custom [`QuantumModel`](advanced_tutorials/custom-models.md)s

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
* `emu-c`: install the Pasqal circuit tensor network emulator EMU-C
* `pulser`: install the Pulser backend. Pulser is a framework for composing, simulating and executing pulse sequences for neutral-atom quantum devices.
* `visualization`: install the library necessary to visualize the quantum circuits in SVG.

!!! warning
    In order to correctly install the "visualization" extra, you need to have Cairo installed in your system. This
    depends on the operating system you are using:

    ```bash
    # on Ubuntu
    sudo apt install pkg-config libcairo2-dev

    # on MacOS
    brew install pkg-config cairo

    # or with conda
    conda install pycairo
    ```
---
