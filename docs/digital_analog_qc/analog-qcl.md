## Fitting a simple function

!!! warning
    Tutorial to be updated

Analog blocks can be parametrized in the usual Qadence manner. Like any other parameters, they can be optimized. The next snippet examplifies the creation of an analog and paramertized ansatze to fit a sine function. First, define an ansatz block and an observable:

```python exec="on" source="material-block" session="sin"
import torch
from qadence import Register, FeatureParameter, VariationalParameter
from qadence import AnalogRX, AnalogRZ, Z
from qadence import wait, chain, add

pi = torch.pi

# A two qubit register.
reg = Register.from_coordinates([(0, 0), (0, 12)])

# An analog ansatz with an input time parameter.
t = FeatureParameter("t")

block = chain(
    AnalogRX(pi/2.),
    AnalogRZ(t),
    wait(1000 * VariationalParameter("theta", value=0.5)),
    AnalogRX(pi/2),
)

# Total magnetization observable.
obs = add(Z(i) for i in range(reg.n_qubits))
```

??? note "Plotting functions `plot` and `scatter`"
    ```python exec="on" session="sin" source="material-block"
    def plot(ax, x, y, **kwargs):
        xnp = x.detach().cpu().numpy().flatten()
        ynp = y.detach().cpu().numpy().flatten()
        ax.plot(xnp, ynp, **kwargs)

    def scatter(ax, x, y, **kwargs):
        xnp = x.detach().cpu().numpy().flatten()
        ynp = y.detach().cpu().numpy().flatten()
        ax.scatter(xnp, ynp, **kwargs)
    ```

Next, define the dataset to train on and plot the initial prediction. The differentiation mode can be set to either `DiffMode.AD` or `DiffMode.GPSR`.

```python exec="on" source="material-block" html="1" result="json" session="sin"
import matplotlib.pyplot as plt
from qadence import QuantumCircuit, QuantumModel, DiffMode

# Define a quantum model including digital-analog emulation.
circ = QuantumCircuit(reg, block)
model = QuantumModel(circ, obs, diff_mode=DiffMode.GPSR)

# Time support dataset.
x_train = torch.linspace(0, 6, steps=30)
# Function to fit.
y_train = -0.64 * torch.sin(x_train + 0.33) + 0.1
# Initial prediction.
y_pred_initial = model.expectation({"t": x_train})

fig, ax = plt.subplots() # markdown-exec: hide
plt.xlabel("Time [μs]") # markdown-exec: hide
plt.ylabel("Sin [arb.]") # markdown-exec: hide
scatter(ax, x_train, y_train, label="Training points", marker="o", color="green") # markdown-exec: hide
plot(ax, x_train, y_pred_initial, label="Initial prediction") # markdown-exec: hide
plt.legend() # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
```

Finally, the classical optimization part is handled by PyTorch:

```python exec="on" source="material-block" html="1" result="json" session="sin"

# Use PyTorch built-in functionality.
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

# Define a loss function.
def loss_fn(x_train, y_train):
    return mse_loss(model.expectation({"t": x_train}).squeeze(), y_train)

# Number of epochs to train over.
n_epochs = 200

# Optimization loop.
for i in range(n_epochs):
    optimizer.zero_grad()

    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()

# Get and visualize the final prediction.
y_pred = model.expectation({"t": x_train})

fig, ax = plt.subplots() # markdown-exec: hide
plt.xlabel("Time [μs]") # markdown-exec: hide
plt.ylabel("Sin [arb.]") # markdown-exec: hide
scatter(ax, x_train, y_train, label="Training points", marker="o", color="green") # markdown-exec: hide
plot(ax, x_train, y_pred_initial, label="Initial prediction") # markdown-exec: hide
plot(ax, x_train, y_pred, label="Final prediction") # markdown-exec: hide
plt.legend() # markdown-exec: hide
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(fig)) # markdown-exec: hide
assert loss_fn(x_train, y_train) < 0.05 # markdown-exec: hide
```
