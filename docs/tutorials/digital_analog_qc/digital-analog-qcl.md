In the [analog QCL tutorial](analog-blocks-qcl.md) we used analog blocks to learn a function of interest. The analog blocks are a direct abstraction of device execution with global addressing. However, we may want to directly program an Hamiltonian-level ansatz to have a finer control on our model. In Qadence this can easily be done through digital-analog programs. In this tutorial we will solve a simple QCL problem with this approach.

## Setting up the problem

```python exec="on" source="material-block" html="1" session="da-qcl"
from docs import docsutils # markdown-exec: hide
import torch
import matplotlib.pyplot as plt

# Function to fit:
def f(x):
    return x**5

xmin = -1.0
xmax = 1.0
n_test = 100

x_test = torch.linspace(xmin, xmax, steps = n_test)
y_test = f(x_test)

plt.clf()  # markdown-exec: hide
plt.plot(x_test, y_test)
plt.xlim((-1.1, 1.1))
plt.ylim((-1.1, 1.1))
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Digital-Analog Ansatz

We start by defining the register of qubits. The topology we use now will define the interactions in the entangling Hamiltonian. As an example, we can define a rectangular lattice with 6 qubits.

```python exec="on" source="material-block" session="da-qcl"
from qadence import Register

reg = Register.rectangular_lattice(
    qubits_row = 3,
    qubits_col = 2,
)

reg.draw()
```

Inspired by the Ising interaction mode of Rydberg atoms, we can now define the interaction Hamiltonian as $\mathcal{H}_{ij}=\frac{1}{r_{ij}^6}N_iN_j$ where $N_i=(1/2)(I_i-Z_i)$ is the number operator and and $r_{ij}$ is the distance between qubits $i$ and $j$. We can easily instatiate this interaction Hamiltonian from the register information:

```python exec="on" source="material-block" session="da-qcl"
from qadence import N, add

def h_ij(i: int, j: int):
    return N(i)@N(j)

h_int = add(h_ij(*edge)/r**6 for edge, r in reg.edge_distances.items())
```

To build the digital-analog ansatz we can make use of the standard `hea` function by specifying we want to use the `Strategy.SDAQC` and passing the Hamiltonian we created as the entangler. The entangling operation will thus be the evolution of this Hamiltonian `HamEvo(h_int, t)`, where the time parameter `t` is considered to be a variational parameter at each layer.

```python exec="on" source="material-block" html=1 session="da-qcl"
from qadence import hea, Strategy, RX, RY
from qadence.draw import html_string # markdown-exec: hide

depth = 2

da_ansatz = hea(
    n_qubits = reg.n_qubits,
    depth = depth,
    operations = [RX, RY, RX],
    entangler = h_int,
    strategy = Strategy.SDAQC,
)

print(html_string(da_ansatz))
```

## Creating the QuantumModel

The rest of the procedure is the same as any other Qadence workflow. We start by defining a feature map for input encoding and an observable for output decoding.

```python exec="on" source="material-block" session="da-qcl"
from qadence import feature_map, BasisSet, ReuploadScaling
from qadence import Z, I

fm = feature_map(
    n_qubits = reg.n_qubits,
    param = "x",
    fm_type = BasisSet.CHEBYSHEV,
    reupload_scaling = ReuploadScaling.TOWER,
)

# Total magnetization
observable = add(Z(i) for i in range(reg.n_qubits))
```

And we have all the ingredients to initialize the `QuantumModel`:

```python exec="on" source="material-block" session="da-qcl"
from qadence import QuantumCircuit, QuantumModel

circuit = QuantumCircuit(reg, fm, da_ansatz)

model = QuantumModel(circuit, observable = observable)
```

## Training the model

We can now train the model. As an example we create a set of twenty training points

```python exec="on" source="material-block" session="da-qcl"
# Chebyshev FM does not accept x = -1, 1
xmin = -0.99
xmax = 0.99
n_train = 20

x_train = torch.linspace(xmin, xmax, steps = n_train)
y_train = f(x_train)

# Initial model prediction
y_pred_initial = model.expectation({"x": x_test}).detach()
```

```python exec="on" source="material-block" session="da-qcl"
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

n_epochs = 200

def loss_fn(x_train, y_train):
    out = model.expectation({"x": x_train})
    loss = criterion(out.squeeze(), y_train)
    return loss

for i in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn(x_train, y_train)
    loss.backward()
    optimizer.step()
```

```python exec="on" source="material-block" html="1" session="da-qcl"
y_pred_final = model.expectation({"x": x_test}).detach()

plt.clf()  # markdown-exec: hide
plt.plot(x_test, y_pred_initial, label = "Initial prediction")
plt.plot(x_test, y_pred_final, label = "Final prediction")
plt.scatter(x_train, y_train, label = "Training points")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim((-1.1, 1.1))
plt.ylim((-1.1, 1.1))
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```
