Analog blocks can be parametrized in the usual Qadence manner. Like any other parameters,
they can be optimized. The next snippet examplifies the creation of an analog and parameterized ansatz
to fit a simple function. First, define a register and feature map block. We again use a default spacing of
$8~\mu\text{m}$ as done in the [basic tutorial](analog-basics.md).


```python exec="on" source="material-block" session="qcl"
from qadence import Register, FeatureParameter, chain
from qadence import AnalogRX, AnalogRY, AnalogRZ, wait
from sympy import acos

# Line register
n_qubits = 2
register = Register.line(n_qubits, spacing = 8.0)

# The input feature x for the circuit to learn f(x)
x = FeatureParameter("x")

# Feature map with a few global analog rotations
fm = chain(
    AnalogRX(x),
    AnalogRY(2*x),
    AnalogRZ(3*x),
)
```

Next, we define the ansatz with parameterized rotations.

```python exec="on" source="material-block" session="qcl"
from qadence import hamiltonian_factory, Z
from qadence import QuantumCircuit, QuantumModel, BackendName, DiffMode
from qadence import VariationalParameter

t_0 = 1000. * VariationalParameter("t_0")
t_1 = 1000. * VariationalParameter("t_1")
t_2 = 1000. * VariationalParameter("t_2")

# Creating the ansatz with parameterized rotations and wait time
ansatz = chain(
    AnalogRX("tht_0"),
    AnalogRY("tht_1"),
    AnalogRZ("tht_2"),
    wait(t_0),
    AnalogRX("tht_3"),
    AnalogRY("tht_4"),
    AnalogRZ("tht_5"),
    wait(t_1),
    AnalogRX("tht_6"),
    AnalogRY("tht_7"),
    AnalogRZ("tht_8"),
    wait(t_2),
)
```

We define the measured observable as the total magnetization, and build the `QuantumModel`.

```python exec="on" source="material-block" session="qcl"
# Total magnetization observable
observable = hamiltonian_factory(n_qubits, detuning = Z)

# Defining the circuit and observable
circuit = QuantumCircuit(register, fm, ansatz)

model = QuantumModel(
    circuit,
    observable = observable,
    backend = BackendName.PYQTORCH,
    diff_mode = DiffMode.AD
)
```

Now we can define the function to fit as well as our training and test data.

```python exec="on" source="material-block" session="qcl"
import torch
import matplotlib.pyplot as plt

# Function to fit:
def f(x):
    return x**2

x_test = torch.linspace(-1.0, 1.0, steps=100)
y_test = f(x_test)

x_train = torch.linspace(-1.0, 1.0, steps=10)
y_train = f(x_train)

# Initial prediction from the model, to be visualized later
y_pred_initial = model.expectation({"x": x_test}).detach()
```

Finally we define a simple loss function and training loop.

```python exec="on" source="material-block" session="qcl"
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def loss_fn(x_train, y_train):
    out = model.expectation({"x": x_train})
    loss = mse_loss(out.squeeze(), y_train)
    return loss

n_epochs = 200

for i in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()
```

And with the model trained we can plot the final results.

```python exec="on" source="material-block" html="1" session="qcl"
y_pred_final = model.expectation({"x": x_test}).detach()

plt.clf()  # markdown-exec: hide
plt.plot(x_test, y_pred_initial, label = "Initial prediction")
plt.plot(x_test, y_pred_final, label = "Final prediction")
plt.scatter(x_train, y_train, label = "Training points")
plt.xlabel("Feature x")  # markdown-exec: hide
plt.ylabel("f(x)")  # markdown-exec: hide
plt.xlim((-1.1, 1.1))  # markdown-exec: hide
plt.ylim((-0.1, 1.1))  # markdown-exec: hide
plt.legend()  # markdown-exec: hide

from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```
