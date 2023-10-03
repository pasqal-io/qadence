# Digital-Analog Emulation

!!! note "TL;DR: Automatic emulation in the `pyqtorch` backend"

    All analog blocks are automatically translated to their emulated version when running them
    with the `pyqtorch` backend (by calling `add_interaction` on them under the hood):

    ```python exec="on" source="material-block" result="json"
    import torch
    from qadence import Register, AnalogRX, sample

    reg = Register.from_coordinates([(0,0), (0,5)])
    print(sample(reg, AnalogRX(torch.pi)))
    ```


Qadence includes primitives for the simple construction of ising-like
Hamiltonians to account for the interaction among qubits.  This allows to
simulate systems closer to real quantum computing platforms such as
neutral atoms. The constructed Hamiltonians are of the form

$$
\mathcal{H} = \sum_{i} \frac{\hbar\Omega}{2} \hat\sigma^x_i - \sum_{i} \hbar\delta \hat n_i  + \mathcal{H}_{int},
$$


where $\hat n = \frac{1-\hat\sigma_z}{2}$, and $\mathcal{H}_{int}$ is a pair-wise interaction term.


We currently have two central operations that can be used to compose analog programs.

- [`WaitBlock`][qadence.blocks.analog.WaitBlock] for interactions
- [`ConstantAnalogRotation`][qadence.blocks.analog.ConstantAnalogRotation]

Both are _time-independent_ and can be emulated by calling `add_interaction`.

To compose analog blocks you can use `chain` and `kron` as usual with the following restrictions:

- [`AnalogChain`][qadence.blocks.analog.AnalogChain]s can only be constructed from AnalogKron blocks
  or _**globally supported**_ primitive, analog blocks.
- [`AnalogKron`][qadence.blocks.analog.AnalogKron]s can only be constructed from _**non-global**_,
  analog blocks with the _**same duration**_.

The `wait` operation can be emulated with an *Ising* or an $XY$-interaction:

```python exec="on" source="material-block" result="json"
from qadence import Register, wait, add_interaction, run

block = wait(duration=3000)
print(block)

print("") # markdown-exec: hide
reg = Register.from_coordinates([(0,0), (0,5)])  # we need atomic distances
emulated = add_interaction(reg, block, interaction="XY")  # or: interaction="Ising"
print(emulated.generator)
```


The `AnalogRot` constructor can create any constant (in time), analog rotation.

```python exec="on" source="material-block" result="json"
import torch
from qadence import AnalogRot, AnalogRX

# implement a global RX rotation
block = AnalogRot(
    duration=1000.,  # [ns]
    omega=torch.pi, # [rad/μs]
    delta=0,        # [rad/μs]
    phase=0,        # [rad]
)
print(block)

# or use the short hand
block = AnalogRX(torch.pi)
print(block)
```

Analog blocks can also be `chain`ed, and `kron`ed like all other blocks, but with two small caveats:

```python exec="on" source="material-block"
import torch
from qadence import AnalogRot, kron, chain, wait

# only blocks with the same `duration` can be `kron`ed
kron(
    wait(duration=1000, qubit_support=(0,1)),
    AnalogRot(duration=1000, omega=2.0, qubit_support=(2,3))
)

# only blocks with `"global"` or the same qubit support can be `chain`ed
chain(wait(duration=200), AnalogRot(duration=300, omega=2.0))
```

!!! note "Composing digital & analog blocks"
    You can also compose digital and analog blocks where the additional restrictions of `chain`/`kron`
    only apply to composite blocks which only contain analog blocks. For more details/examples, see
    [`AnalogChain`][qadence.blocks.analog.AnalogChain] and [`AnalogKron`][qadence.blocks.analog.AnalogKron].


## Fitting a simple function

Just as most other blocks, analog blocks can be parametrized, and thus we can build a
small ansatz which can fit a sine wave. When using the `pyqtorch` backend the
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
