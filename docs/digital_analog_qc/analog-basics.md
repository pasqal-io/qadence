!!! note
    The digital-analog emulation framework is under construction and significant changes to the interface
    should be expected in the near-future. Nevertheless, the currest version serves as a prototype of the
    functionality, and any feedback is greatly appreciated.

Qadence includes primitives for the construction of programs implemented on a set of interacting qubits.
The goal is to build digital-analog programs that better represent the reality of interacting qubit
platforms, such as neutral-atoms, while maintaining a simplified interface for users coming from
a digital quantum computing background that may not be as familiar with pulse-level programming.

To build the intuition for the interface in Qadence, it is important to go over some of the underlying physics.
We can write a general Hamiltonian for a set of $n$ interacting qubits as

$$
\mathcal{H} = \sum_{i=0}^{n-1}\left(\mathcal{H}^\text{d}_{i}(t) + \sum_{j<i}\mathcal{H}^\text{int}_{ij}\right),
$$

where the *driving Hamiltonian* $\mathcal{H}^\text{d}_{i}$ describes the pulses used to control single-qubit rotations,
and the *interaction Hamiltonian* $\mathcal{H}^\text{int}_{ij}$ describes the natural interaction between qubits.

## Rydberg atoms

For the purpose of digital-analog emulation of neutral-atom systems in Qadence, we now consider a
simplified **time-independent global** driving Hamiltonian, written as

$$
\mathcal{H}^\text{d}_{i} = \frac{\Omega}{2}\left(\cos(\phi) X_i - \sin(\phi) Y_i \right) - \delta N_i
$$

where $\Omega$ is the Rabi frequency, $\delta$ is the detuning, $\phi$ is the phase, $X_i$ and $Y_i$ are the standard
Pauli operators, and $N_i=\frac{1}{2}(I_i-Z_i)$ is the number operator. This Hamiltonian allows arbitrary global single-qubit
rotations to be written, meaning that the values set for $(\Omega,\phi,\delta)$ are the same accross the qubit support.

For the interaction term, Rydberg atoms typically allow both an Ising and an XY mode of operation. For now, we focus on
the Ising interaction, where the Hamiltonian is written as

$$
\mathcal{H}^\text{int}_{ij} = \frac{C_6}{r_{ij}^6}N_iN_j
$$

where $r_{ij}$ is the distance between atoms $i$ and $j$, and $C_6$ is a coefficient depending on the specific Rydberg level
of the excited state used in the computational logic states.

For a given register of atoms prepared in some spatial coordinates, the Hamiltonians described will generate the dynamics
of some unitary operation as

$$
U(t, \Omega, \delta, \phi) = \exp(-i\mathcal{H}t)
$$

where we specify the final parameter $t$, the duration of the operation.

Qadence uses the following units for user-specified parameters:

- Rabi frequency and detuning $\Omega$, $\delta$: $[\text{rad}/\mu \text{s}]$
- Phase $\phi$: $[\text{rad}]$
- Duration $t$: $[\text{ns}]$
- Atomic coordinates: $[\mu \text{m}]$


## In practice

Given the Hamiltonian description in the previous section, we will now go over a
few examples of the standard operations available in Qadence.

### Arbitrary rotation

To start, we will exemplify the a general rotation on a set of atoms. To create an arbitrary
register of atoms, we refer the user to the [register creation tutorial](../tutorials/register.md).
Note that, for now, we do not use any information regarding the edges of the register graph, only
the coordinates of each node that are used to compute the distance $r_{ij}$ in the interaction term.
Below, we create a line register of three qubits.

!!! note
    For now we will create registers directly from the coordinates to maintain full control over
    the coordinates and spacing between the atoms. This avoids inconsistencies on how register spacings
    are passed to both the pyqtorch and Pulser backends. The interface will be unified soon.

```python exec="on" source="material-block" session="emu"
from qadence import Register

dx = 8.0

reg = Register.from_coordinates([(dx, 0), (2*dx, 0), (3*dx, 0)])
```

Currently, the most general rotation operation uses the `AnalogRot` operation, which
essentially implements $U(t, \Omega, \delta, \phi)$ defined above.

```python exec="on" source="material-block" session="emu"
import torch
from qadence import AnalogRot

rot_op = AnalogRot(
    duration = 1000., # [ns]
    omega = torch.pi, # [rad/μs]
    delta = torch.pi, # [rad/μs]
    phase = torch.pi, # [rad]
)
```

Note that in the code above a specific qubit support is not defined. By default this operation
applies a *global* rotation on all qubits. We can define a circuit using the 3-qubit register
and run it in the pyqtorch backend:

```python exec="on" source="material-block" result="json" session="emu"
from qadence import QuantumCircuit, QuantumModel, BackendName

circuit = QuantumCircuit(reg, rot_op)
model = QuantumModel(circuit, backend = BackendName.PYQTORCH)
wf = model.run()

print(wf)
```

<details>
    <summary>Under the hood of AnalogRot</summary>

    To be fully explicit about what goes on under the hood of `AnalogRot`, we can look at the example
    code below.

    ```python exec="on" source="material-block" result="json"
    from qadence import QuantumCircuit, QuantumModel, BackendName
    from qadence import HamEvo, X, Y, N, add
    from qadence.analog.utils import C6_DICT
    import math
    import torch

    # Following the 3-qubit register above
    n_qubits = 3
    dx = 8.0

    # Parameters used in the AnalogRot
    duration = 1000.
    omega = torch.pi
    delta = torch.pi
    phase = torch.pi

    # Building the terms in the driving Hamiltonian
    h_x = (omega / 2) * math.cos(phase) * add(X(i) for i in range(n_qubits))
    h_y = (-1.0 * omega / 2) * math.sin(phase) * add(Y(i) for i in range(n_qubits))
    h_n = -1.0 * delta * add(N(i) for i in range(n_qubits))

    # Building the interaction Hamiltonian

    # Dictionary of coefficient values for each Rydberg level, which is 60 by default
    c_6 = C6_DICT[60]

    h_int = c_6 * (
        1/(dx**6) * (N(0)@N(1)) +
        1/(dx**6) * (N(1)@N(2)) +
        1/((2*dx)**6) * (N(0)@N(2))
    )

    h_d = h_x + h_y + h_n + h_int

    # Convert duration to µs due to the units of the Hamiltonian
    explicit_rot = HamEvo(h_d, duration / 1000)

    circuit = QuantumCircuit(n_qubits, explicit_rot)
    model = QuantumModel(circuit, backend = BackendName.PYQTORCH)
    wf = model.run()

    # We get the same final wavefunction
    print(wf)
    ```
</details>

When sending the `AnalogRot` operation to the pyqtorch backend, Qadence
automatically builds the correct Hamiltonian and the corresponding `HamEvo`
operation with the added qubit interactions, as shown explicitly in the
minimized section above. However, this operation is also supported in the
Pulser backend, where the correct pulses are automatically created.

!!! warning
    When using the Pulser backend it is currently advised to always explicitly pass
    the register spacing in the `configuration` dictionary, which is a constant that
    multiplies the coordinates of the register. The passing of register spacing
    will soon be unified.


```python exec="on" source="material-block" result="json" session="emu"
from qadence import DiffMode

diff_mode = DiffMode.GPSR  # We have to explicitly change the diff mode for the pulser backend
config = {"spacing": 1.0}  # This ensures the register passed to Pulser is not re-scaled

model = QuantumModel(
    circuit,
    backend = BackendName.PULSER,
    diff_mode = diff_mode,
    configuration = config
    )

wf = model.run()

print(wf)
```

### RX / RY / RZ rotations

The `AnalogRot` provides full control over the parameters of $\mathcal{H}^\text{d}$, but users coming from
a digital quantum computing background may be more familiar with the standard `RX`, `RY` and `RZ` rotations, also available in Qadence. For the emulated analog interface, Qadence provides alternative
`AnalogRX`, `AnalogRY` and `AnalogRZ` operations which call `AnalogRot` under the hood to represent
the rotations accross the respective axes.

For a given angle of rotation $\theta$ provided to each of these operations, currently a set of hardcoded assumptions are made on the tunable Hamiltonian parameters:

$$
\begin{aligned}
\text{RX}:& \quad \Omega = \pi, \quad \delta = 0, \quad \phi = 0, \quad t = (\theta/\Omega)\times 10^3 \\
\text{RY}:& \quad \Omega = \pi, \quad \delta = 0, \quad \phi = -\pi/2, \quad t = (\theta/\Omega)\times 10^3 \\
\text{RZ}:& \quad \Omega = 0, \quad \delta = \pi, \quad \phi = 0, \quad t = (\theta/\delta)\times 10^3 \\
\end{aligned}
$$

Note that the $\text{RZ}$ operation as defined above includes a global phase compared to the
standard $\text{RZ}$ rotation since it evolves $\exp\left(-i\frac{\theta}{2}\frac{I-Z}{2}\right)$ instead of $\exp\left(-i\frac{\theta}{2}Z\right)$ given the detuning operator in $\mathcal{H}^\text{d}$.

!!! note
    As shown above, the values of $\Omega$ and $\delta$ are hardcoded in these operators, and the
    effective angle of rotations is controlled by varying the duration of the evolution. Currently,
    the best way to overcome this is to use `AnalogRot` directly, but a more convenient interface
    will be provided soon.

Below we exemplify the usage of `AnalogRX`

```python exec="on" source="material-block" result="json" session="rx"
import torch

from qadence import Register, QuantumCircuit, QuantumModel, BackendName, DiffMode
from qadence import AnalogRX, random_state, equivalent_state, kron, RX

dx = 8.0

reg = Register.from_coordinates([(dx, 0), (2*dx, 0), (3*dx, 0)])
n_qubits = 3

# Rotation angle
theta = torch.pi

# Analog rotation using the Rydberg Hamiltonian
rot_analog = AnalogRX(angle = theta)

# Equivalent full-digital global rotation
rot_digital = kron(RX(i, theta) for i in range(n_qubits))

circuit_analog = QuantumCircuit(reg, rot_analog)
circuit_digital = QuantumCircuit(reg, rot_digital)

model_analog_pyq = QuantumModel(circuit_analog, backend = BackendName.PYQTORCH)
model_digital_pyq = QuantumModel(circuit_digital, backend = BackendName.PYQTORCH)

# Some random initial state
init_state = random_state(n_qubits)

# Compare the final state using the full digital and the AnalogRX
wf_analog_pyq = model_analog_pyq.run(state = init_state)
wf_digital_pyq = model_digital_pyq.run(state = init_state)

bool_equiv = equivalent_state(wf_analog_pyq, wf_digital_pyq, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

As we can see, running a global `RX` or the `AnalogRX` does not result in equivalent states at the end, given
that the digital `RX` operation does not include the interaction between the qubits. By setting `dx` very high
in the code above the interaction will be less significant and the results will match.

However, if we compare with the Pulser backend, we see that the results for `AnalogRX` are consistent with
the expected results from a real device:

```python exec="on" source="material-block" result="json" session="rx"
config = {"spacing": 1.0}

model_analog_pulser = QuantumModel(
    circuit_analog,
    backend = BackendName.PULSER,
    diff_mode = DiffMode.GPSR,
    configuration = config
)

wf_analog_pulser = model_analog_pulser.run(state = init_state)

bool_equiv = equivalent_state(wf_analog_pyq, wf_analog_pulser, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

### Evolving the interaction term

Finally, besides applying specific qubit rotations, we can also choose to evolve only the interaction term
$\mathcal{H}^\text{int}$, equivalent to setting $\Omega = \delta = \phi = 0$. To do so, Qadence provides the
function `wait` which does exactly this.

```python exec="on" source="material-block" result="json" session="rx"
from qadence import wait, run

dx = 8.0
reg = Register.from_coordinates([(dx, 0), (2*dx, 0), (3*dx, 0)])
n_qubits = 3

duration = 1000.
op = wait(duration = duration)

init_state = random_state(n_qubits)

wf_pyq = run(reg, op, state = init_state, backend = BackendName.PYQTORCH)
wf_pulser = run(reg, op, state = init_state, backend = BackendName.PULSER, configuration = {"spacing": 1.0})

bool_equiv = equivalent_state(wf_pyq, wf_pulser, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

## Some technical details

To be added.

<!-- PREVIOUS STUFF, KEPT TEMPORARILY:

- [`WaitBlock`][qadence.blocks.analog.WaitBlock] by free-evolving $\mathcal{H}_{\textrm{int}}$
- [`ConstantAnalogRotation`][qadence.blocks.analog.ConstantAnalogRotation] by free-evolving $\mathcal{H}$

The `wait` operation can be emulated with an $ZZ$- (Ising) or an $XY$-interaction:

```python exec="on" source="material-block" result="json"
from qadence import Register, wait, add_interaction, run, Interaction

block = wait(duration=3000)
print(f"block = {block} \n") # markdown-exec: hide

reg = Register.from_coordinates([(0,0), (0,5)])  # Dimensionless.
emulated = add_interaction(reg, block, interaction=Interaction.XY)  # or Interaction.ZZ for Ising.

print("emulated.generator = \n") # markdown-exec: hide
print(emulated.generator) # markdown-exec: hide
```

The `AnalogRot` constructor can be used to create a fully customizable `ConstantAnalogRotation` instances:

```python exec="on" source="material-block" result="json"
import torch
from qadence import AnalogRot, AnalogRX

# Implement a global RX rotation by setting all parameters.
block = AnalogRot(
    duration=1000., # [ns]
    omega=torch.pi, # [rad/μs]
    delta=0,        # [rad/μs]
    phase=0,        # [rad]
)
print(f"AnalogRot = {block}\n") # markdown-exec: hide

# Or use the shortcut.
block = AnalogRX(torch.pi)
print(f"AnalogRX = {block}") # markdown-exec: hide
```

!!! note "Automatic emulation in the PyQTorch backend"

    All analog blocks are automatically translated to their emulated version when running them
    with the PyQTorch backend:

    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import Register, AnalogRX, sample

    reg = Register.from_coordinates([(0,0), (0,5)])
	sample = sample(reg, AnalogRX(torch.pi))
    print(f"sample = {sample}") # markdown-exec: hide
    ```

To compose analog blocks, the regular `chain` and `kron` operations can be used under the following restrictions:

- The resulting [`AnalogChain`][qadence.blocks.analog.AnalogChain] type can only be constructed from `AnalogKron` blocks
  or _**globally supported**_ primitive analog blocks.
- The resulting [`AnalogKron`][qadence.blocks.analog.AnalogKron] type can only be constructed from _**non-global**_
  analog blocks with the _**same duration**_.

```python exec="on" source="material-block" result="json"
import torch
from qadence import AnalogRot, kron, chain, wait

# Only analog blocks with a global qubit support can be composed
# using chain.
analog_chain = chain(wait(duration=200), AnalogRot(duration=300, omega=2.0))
print(f"Analog Chain block = {analog_chain}") # markdown-exec: hide

# Only blocks with the same `duration` can be composed using kron.
analog_kron = kron(
    wait(duration=1000, qubit_support=(0,1)),
    AnalogRot(duration=1000, omega=2.0, qubit_support=(2,3))
)
print(f"Analog Kron block = {analog_kron}") # markdown-exec: hide
```

!!! note "Composing digital & analog blocks"
    It is possible to compose digital and analog blocks where the additional restrictions for `chain` and `kron`
    only apply to composite blocks which contain analog blocks only. For further details, see
    [`AnalogChain`][qadence.blocks.analog.AnalogChain] and [`AnalogKron`][qadence.blocks.analog.AnalogKron]. -->
