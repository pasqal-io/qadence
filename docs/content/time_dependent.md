For use cases when the Hamiltonian of the system is time-dependent, Qadence provides a special parameter `TimePrameter("t")` that denotes the explicit time dependence. Using this time parameter one can define a parameterized block acting as the generator passed to `HamEvo` that encapsulates the required time dependence function.

```python exec="on" source="material-block" session="getting_started" result="json"
from qadence import X, Y, HamEvo, TimeParameter, Parameter, run
from pyqtorch.utils import SolverType
import torch

# simulation parameters
duration = 1.0  # duration of time-dependent block simulation
ode_solver = SolverType.DP5_SE  # time-dependent Schrodinger equation solver method
n_steps_hevo = 500  # integration time steps used by solver

# define block parameters
t = TimeParameter("t")
omega_param = Parameter("omega")

# create time-dependent generator
generator_td = omega_param * (t * X(0) + t**2 * Y(1))

# create parameterized HamEvo block
hamevo = HamEvo(generator_td, 0.0, duration=duration)

# run simulation
out_state = run(hamevo,
                values={"omega": torch.tensor(10.0)},
                configuration={"ode_solver": ode_solver,
                               "n_steps_hevo": n_steps_hevo})

print(out_state)
```

Note that when using `HamEvo` with a time-dependent generator, its second argument `parameter` is not used and an arbitrary value can be passed to it. However, in case of time-dependent generator a value for `duration` argument to `HamEvo` must be passed in order to define the duration of the simulation. The unit of passed duration value $\tau$ must be aligned with the units of other parameters in the time-dependent generator so that the integral of generator $\overset{\tau}{\underset{0}{\int}}\mathcal{\hat{H}}(t){\rm d}t$ is dimensionless.
