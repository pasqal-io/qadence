!!! warning
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
- Atom coordinates: $[\mu \text{m}]$


## In practice

Given the Hamiltonian description in the previous section, we will now go over a
few examples of the standard operations available in Qadence.

### Arbitrary rotation

To start, we will exemplify the a general rotation on a set of atoms. To create an arbitrary
register of atoms, we refer the user to the [register creation tutorial](../tutorials/register.md).
In this tutorial we do not use any information regarding the edges of the register graph, only
the coordinates of each node that are used to compute the distance $r_{ij}$ in the interaction term.
Below, we create a line register of three qubits directly from the coordinates.

```python exec="on" source="material-block" session="emu"
from qadence import Register

dx = 8.0  # Atom spacing in μm
reg = Register.from_coordinates([(0, 0), (dx, 0), (2*dx, 0)])
```

Currently, the most general rotation operation uses the `AnalogRot` operation, which
essentially implements $U(t, \Omega, \delta, \phi)$ defined above.

```python exec="on" source="material-block" session="emu"
from math import pi
from qadence import AnalogRot

rot_op = AnalogRot(
    duration = 500., # [ns]
    omega = pi, # [rad/μs]
    delta = pi, # [rad/μs]
    phase = pi, # [rad]
)
```

Note that in the code above a specific qubit support is not defined. By default this operation
applies a *global* rotation on all qubits. We can define a circuit using the 3-qubit register
and run it in the pyqtorch backend:

```python exec="on" source="material-block" result="json" session="emu"
from qadence import BackendName, run

wf = run(reg, rot_op, backend = BackendName.PYQTORCH)

print(wf)
```

<details>
    <summary>Under the hood of AnalogRot</summary>

    To be fully explicit about what goes on under the hood of `AnalogRot`, we can look at the example
    code below.

    ```python exec="on" source="material-block" result="json"
    from qadence import BackendName, HamEvo, X, Y, N, add, run
    from qadence.analog.utils import C6_DICT
    from math import pi, cos, sin

    # Following the 3-qubit register above
    n_qubits = 3
    dx = 8.0

    # Parameters used in the AnalogRot
    duration = 500.
    omega = pi
    delta = pi
    phase = pi

    # Building the terms in the driving Hamiltonian
    h_x = (omega / 2) * cos(phase) * add(X(i) for i in range(n_qubits))
    h_y = (-1.0 * omega / 2) * sin(phase) * add(Y(i) for i in range(n_qubits))
    h_n = -1.0 * delta * add(N(i) for i in range(n_qubits))

    # Building the interaction Hamiltonian

    # Dictionary of coefficient values for each Rydberg level, which is 60 by default
    c_6 = C6_DICT[60]

    h_int = c_6 * (
        1/(dx**6) * (N(0)@N(1)) +
        1/(dx**6) * (N(1)@N(2)) +
        1/((2*dx)**6) * (N(0)@N(2))
    )

    hamiltonian = h_x + h_y + h_n + h_int

    # Convert duration to µs due to the units of the Hamiltonian
    explicit_rot = HamEvo(hamiltonian, duration / 1000)

    wf = run(n_qubits, explicit_rot, backend = BackendName.PYQTORCH)

    # We get the same final wavefunction
    print(wf)
    ```
</details>

When sending the `AnalogRot` operation to the pyqtorch backend, Qadence
automatically builds the correct Hamiltonian and the corresponding `HamEvo`
operation with the added qubit interactions, as shown explicitly in the
minimized section above. However, this operation is also supported in the
Pulser backend, where the correct pulses are automatically created.


```python exec="on" source="material-block" result="json" session="emu"

wf = run(
    reg,
    rot_op,
    backend = BackendName.PULSER,
)

print(wf)
```

### RX / RY / RZ rotations

The `AnalogRot` provides full control over the parameters of $\mathcal{H}^\text{d}$, but users coming from
a digital quantum computing background may be more familiar with the standard `RX`, `RY` and `RZ` rotations,
also available in Qadence. For the emulated analog interface, Qadence provides alternative
`AnalogRX`, `AnalogRY` and `AnalogRZ` operations which call `AnalogRot` under the hood to represent
the rotations accross the respective axis.

For a given angle of rotation $\theta$ provided to each of these operations, currently a set of hardcoded
assumptions are made on the tunable Hamiltonian parameters:

$$
\begin{aligned}
\text{RX}:& \quad \Omega = \pi, \quad \delta = 0, \quad \phi = 0, \quad t = (\theta/\Omega)\times 10^3 \\
\text{RY}:& \quad \Omega = \pi, \quad \delta = 0, \quad \phi = -\pi/2, \quad t = (\theta/\Omega)\times 10^3 \\
\text{RZ}:& \quad \Omega = 0, \quad \delta = \pi, \quad \phi = 0, \quad t = (\theta/\delta)\times 10^3 \\
\end{aligned}
$$

Note that the $\text{RZ}$ operation as defined above includes a global phase compared to the
standard $\text{RZ}$ rotation since it evolves $\exp\left(-i\frac{\theta}{2}\frac{I-Z}{2}\right)$
instead of $\exp\left(-i\frac{\theta}{2}Z\right)$ given the detuning operator in $\mathcal{H}^\text{d}$.

!!! warning
    As shown above, the values of $\Omega$ and $\delta$ are currently hardcoded in these operators, and the
    effective angle of rotation is controlled by varying the duration of the evolution. Currently,
    the best way to overcome this is to use `AnalogRot` directly, but more general and convenient options
    will be provided soon in an improved interface.

Below we exemplify the usage of `AnalogRX`:

```python exec="on" source="material-block" result="json" session="rx"
from qadence import Register, BackendName
from qadence import RX, AnalogRX, random_state, equivalent_state, kron, run
from math import pi

dx = 8.0

reg = Register.from_coordinates([(0, 0), (dx, 0), (2*dx, 0)])
n_qubits = 3

# Rotation angle
theta = pi

# Analog rotation using the Rydberg Hamiltonian
rot_analog = AnalogRX(angle = theta)

# Equivalent full-digital global rotation
rot_digital = kron(RX(i, theta) for i in range(n_qubits))

# Some random initial state
init_state = random_state(n_qubits)

# Compare the final state using the full digital and the AnalogRX
wf_analog_pyq = run(
    reg,
    rot_analog,
    state = init_state,
    backend = BackendName.PYQTORCH
)


wf_digital_pyq = run(
    reg,
    rot_digital,
    state = init_state,
    backend = BackendName.PYQTORCH
)

bool_equiv = equivalent_state(wf_analog_pyq, wf_digital_pyq, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

As we can see, running a global `RX` or the `AnalogRX` does not result in equivalent states at the end, given
that the digital `RX` operation does not include the interaction between the qubits. By setting `dx` very high
in the code above the interaction will be less significant and the results will match.

However, if we compare with the Pulser backend, we see that the results for `AnalogRX` are consistent with
the expected results from a real device:

```python exec="on" source="material-block" result="json" session="rx"

wf_analog_pulser = run(
    reg,
    rot_analog,
    state = init_state,
    backend = BackendName.PULSER,
)

bool_equiv = equivalent_state(wf_analog_pyq, wf_analog_pulser, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

### Evolving the interaction term

Finally, besides applying specific qubit rotations, we can also choose to evolve only the interaction term
$\mathcal{H}^\text{int}$, equivalent to setting $\Omega = \delta = \phi = 0$. To do so, Qadence provides the
function `wait` which does exactly this.

```python exec="on" source="material-block" result="json" session="int"
from qadence import Register, BackendName, random_state, equivalent_state, wait, run

dx = 8.0
reg = Register.from_coordinates([(0, 0), (dx, 0), (2*dx, 0)])
n_qubits = 3

duration = 1000.
op = wait(duration = duration)

init_state = random_state(n_qubits)

wf_pyq = run(reg, op, state = init_state, backend = BackendName.PYQTORCH)
wf_pulser = run(reg, op, state = init_state, backend = BackendName.PULSER)

bool_equiv = equivalent_state(wf_pyq, wf_pulser, atol = 1e-03)

print("States equivalent: ", bool_equiv)
```

## Some technical details

!!! warning
    The details described here are relevant in the current version but are under
    revision for the next version of the emulated analog interface.

In the previous section we have exemplified the main ingredients of the current user-facing functionalities
of the emulated analog interface, and in the next tutorial on Quantum Circuit Learning we will exmplify its usage
in a simple QML example. Here we specify some extra details of this interface.

In the block system, all the Analog rotation operators initialize a [`ConstantAnalogRotation`][qadence.blocks.analog.ConstantAnalogRotation]
block, while the `wait` operation initializes a [`WaitBlock`][qadence.blocks.analog.WaitBlock]. As we have shown, by default,
these blocks use a global qubit support, which can be passed explicitly by setting `qubit_support = "global"`. However, the blocks do support
local qubit supports, with some constraints. The main constraint is that using `kron` on operators with different durations is not allowed.

```python exec="on" source="material-block" result="json" session="details"
from qadence import AnalogRX, AnalogRY, Register, kron

dx = 8.0
reg = Register.from_coordinates([(0, 0), (dx, 0)])

# Does not work (the angle affects the duration, as seen above):
rot_0 = AnalogRX(angle = 1.0, qubit_support = (0,))
rot_1 = AnalogRY(angle = 2.0, qubit_support = (1,))

try:
    block = kron(rot_0, rot_1)
except ValueError as error:
    print("Error:", error)

# Works:
rot_0 = AnalogRX(angle = 1.0, qubit_support = (0,))
rot_1 = AnalogRY(angle = 1.0, qubit_support = (1,))

block = kron(rot_0, rot_1)
```

Using `chain` is only supported between analog blocks with global qubit support:

```python exec="on" source="material-block" session="details"
from qadence import chain

rot_0 = AnalogRX(angle = 1.0, qubit_support = "global")
rot_1 = AnalogRY(angle = 2.0, qubit_support = "global")

block = chain(rot_0, rot_1)
```

The restrictions above only apply to the analog blocks, and analog and digital blocks can currently be composed.

```python exec="on" source="material-block" session="details"
from qadence import RX

rot_0 = AnalogRX(angle = 1.0, qubit_support = "global")
rot_1 = AnalogRY(angle = 2.0, qubit_support = (0,))
rot_digital = RX(1, 1.0)

block_0 = chain(rot_0, rot_digital)
block_1 = kron(rot_1, rot_digital)
```
