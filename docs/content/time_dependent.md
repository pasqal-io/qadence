For use cases when the Hamiltonian of the system is time-dependent, Qadence provides a special parameter `TimeParameter("t")` that denotes the explicit time dependence. Using this time parameter, one can define a parameterized block acting as the generator passed to `HamEvo` that encapsulates the required time dependence function.

# Noiseless time-dependent Hamiltonian evolution

```python exec="on" source="material-block" session="getting_started"
from qadence import X, Y, HamEvo, TimeParameter, FeatureParameter, run
from pyqtorch.utils import SolverType
import torch

# Simulation parameters
ode_solver = SolverType.DP5_SE  # time-dependent Schrodinger equation solver method
n_steps_hevo = 500  # integration time steps used by solver

# Define block parameters
t = TimeParameter("t")
omega_param = FeatureParameter("omega")

# Arbitrarily compose a time-dependent generator
td_generator = omega_param * (t * X(0) + t**2 * Y(1))

# Create parameterized HamEvo block
hamevo = HamEvo(td_generator, t)
```

Note that when using `HamEvo` with a time-dependent generator, the actual time parameter that was used to construct the generator must be passed for the second argument `parameter`.

By default, the code above will initialize an internal parameter `FeatureParameter("duration")` in the `HamEvo`. Alternatively,
the `duration` argument can be used to rename this parameter, or to pass a fixed value directly. If no fixed value is passed,
it must then be set in the `values` dictionary at runtime.

!!! note "Future improvements"
	Currently it is only possible to pass a single value for the duration, and the only result obtained will be the one
    corresponding to the state at end of the integration. In the future we will change the interface to allow directly passing
    some array of save values to obtain expectation values or statevectors at intermediate steps during the evolution.

```python exec="on" source="material-block" session="getting_started" result="json"

values = {"omega": torch.tensor(10.0), "duration": torch.tensor(1.0)}

config = {"ode_solver": ode_solver, "n_steps_hevo": n_steps_hevo}

out_state = run(hamevo, values = values, configuration = config)

print(out_state)
```

Note that Qadence makes no assumption on units. The unit of passed duration value $\tau$ must be aligned with the units of other parameters in the time-dependent generator so that the integral of generator $\overset{\tau}{\underset{0}{\int}}\mathcal{\hat{H}}(t){\rm d}t$ is dimensionless.

# Noisy time-dependent Hamiltonian evolution

To perform noisy time-dependent Hamiltonian evolution, one needs to pass a list of noise operators to the `noise_operators` argument in `HamEvo`. They correspond to the jump operators used within the time-dependent Schrodinger equation solver method `SolverType.DP5_ME`:

```python exec="on" source="material-block" session="getting_started"
from qadence import X, Y, HamEvo, TimeParameter, FeatureParameter, run
from pyqtorch.utils import SolverType
import torch

# Simulation parameters
ode_solver = SolverType.DP5_ME  # time-dependent Schrodinger equation solver method
n_steps_hevo = 500  # integration time steps used by solver

# Define block parameters
t = TimeParameter("t")
omega_param = FeatureParameter("omega")

# Arbitrarily compose a time-dependent generator
td_generator = omega_param * (t * X(0) + t**2 * Y(1))

# Create parameterized HamEvo block
noise_operators = [X(i) for i in td_generator.qubit_support]
hamevo = HamEvo(td_generator, t, noise_operators = noise_operators)

values = {"omega": torch.tensor(10.0), "duration": torch.tensor(1.0)}

config = {"ode_solver": ode_solver, "n_steps_hevo": n_steps_hevo}

out_state = run(hamevo, values = values, configuration = config)

print(out_state)
```

!!! warning "Noise operators definition"
    Note it is not possible to define `noise_operators` with parametric operators. If you want to do so, we recommend obtaining the tensors via run and set `noise_operators` using `MatrixBlock`. Also, `noise_operators` should have the same or a subset of the qubit support of the `HamEvo` instance.
