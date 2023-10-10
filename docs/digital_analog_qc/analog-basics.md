# Digital-Analog Emulation

## From theory to implementation

Qadence includes primitives for the construction of Ising-like
Hamiltonians to account for custom qubit interaction. This allows to
simulate systems close to real quantum computing platforms such as
neutral atoms. The general form for _time-independent_ Ising Hamiltonians is

$$
\mathcal{H} = \sum_{i} \frac{\hbar\Omega}{2} \hat\sigma^x_i - \sum_{i} \hbar\delta \hat n_i  + \mathcal{H}_{\textrm{int}},
$$

where $\Omega$ is the Rabi frequency, $\delta$ is the detuning, $\hat n = \frac{1-\hat\sigma_z}{2}$ is the number operator, and $\mathcal{H}_{\textrm{int}}$ a pair-wise interaction term. Two central operations implement this Hamiltonian as blocks:

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
    [`AnalogChain`][qadence.blocks.analog.AnalogChain] and [`AnalogKron`][qadence.blocks.analog.AnalogKron].

## Fitting a simple function

Analog blocks can indeed be parametrized to, for instance, create small ansatze to fit a sine function. When using the `pyqtorch` backend the
`add_interaction` function is called automatically. As usual, we can choose which
differentiation backend we want to use: autodiff or parameter shift rule (PSR).

First we define an ansatz block and an observable
```python exec="on" source="material-block" session="sin"
import torch
from qadence import Register, FeatureParameter, VariationalParameter
from qadence import AnalogRX, AnalogRZ, Z
from qadence import wait, chain, add

pi = torch.pi

# two qubit register
reg = Register.from_coordinates([(0, 0), (0, 12)])

# analog ansatz with input parameter
t = FeatureParameter("t")
block = chain(
    AnalogRX(pi / 2),
    AnalogRZ(t),
    wait(1000 * VariationalParameter("theta", value=0.5)),
    AnalogRX(pi / 2),
)

# observable
obs = add(Z(i) for i in range(reg.n_qubits))
```

```python exec="on" session="sin"
def plot(ax, x, y, **kwargs):
    xnp = x.detach().cpu().numpy().flatten()
    ynp = y.detach().cpu().numpy().flatten()
    ax.plot(xnp, ynp, **kwargs)

def scatter(ax, x, y, **kwargs):
    xnp = x.detach().cpu().numpy().flatten()
    ynp = y.detach().cpu().numpy().flatten()
    ax.scatter(xnp, ynp, **kwargs)
```

Then we define the dataset we want to train on and plot the initial prediction.
```python exec="on" source="material-block" html="1" result="json" session="sin"
import matplotlib.pyplot as plt
from qadence import QuantumCircuit, QuantumModel

# define quantum model; including digital-analog emulation
circ = QuantumCircuit(reg, block)
model = QuantumModel(circ, obs, diff_mode="gpsr")

x_train = torch.linspace(0, 6, steps=30)
y_train = -0.64 * torch.sin(x_train + 0.33) + 0.1
y_pred_initial = model.expectation({"t": x_train})

fig, ax = plt.subplots()
scatter(ax, x_train, y_train, label="Training points", marker="o", color="green")
plot(ax, x_train, y_pred_initial, label="Initial prediction")
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

The rest is the usual PyTorch training routine.
```python exec="on" source="material-block" html="1" result="json" session="sin"
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)


def loss_fn(x_train, y_train):
    return mse_loss(model.expectation({"t": x_train}).squeeze(), y_train)


# train
n_epochs = 200

for i in range(n_epochs):
    optimizer.zero_grad()

    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()

    # if (i + 1) % 10 == 0:
    #     print(f"Epoch {i+1:0>3} - Loss: {loss.item()}\n")

# visualize
y_pred = model.expectation({"t": x_train})

fig, ax = plt.subplots()
scatter(ax, x_train, y_train, label="Training points", marker="o", color="green")
plot(ax, x_train, y_pred_initial, label="Initial prediction")
plot(ax, x_train, y_pred, label="Final prediction")
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
assert loss_fn(x_train, y_train) < 0.05 # markdown-exec: hide
```
