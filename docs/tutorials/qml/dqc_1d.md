In this tutorial we will show how to use Qadence to solve a basic 1D Ordinary Differential Equation (ODE) with a QNN using Differentiable Quantum Circuits (DQC) [^1].

Consider the following non-linear ODE and boundary condition:

$$
\frac{df}{dx}= 5\times(4x^3+x^2-2x-\frac12), \qquad f(0)=0
$$

It admits an exact solution:

$$
f(x)=5\times(x^4+\frac13x^3-x^2-\frac12x)
$$

Our goal will be to find this solution for $x\in[-1, 1]$.

```python exec="on" source="material-block" session="dqc"
import torch

def dfdx_equation(x: torch.Tensor) -> torch.Tensor:
    """Derivative as per the equation."""
    return 5*(4*x**3 + x**2 - 2*x - 0.5)
```

For the purpose of this tutorial, we will compute the derivative of the circuit using `torch.autograd`. The point of the DQC algorithm is to use differentiable circuits with parameter shift rules. In Qadence, PSR is implemented directly as custom overrides of the derivative function in the autograd engine, and thus we can later change the derivative method for the model itself if we wish.

```python exec="on" source="material-block" session="dqc"
def calc_deriv(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute a derivative of model that learns f(x), computes df/dx using torch.autograd."""
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs = torch.ones_like(inputs),
        create_graph = True,
        retain_graph = True,
    )[0]
    return grad
```

## Defining the loss function

The essential part of solving this problem is to define the right loss function to represent our goal. In this case, we want to define a model that has the capacity to learn the target solution, and we want to minimize:
- The derivative of this model in comparison with the exact derivative in the equation;
- The output of the model at the boundary in comparison with the value for the boundary condition;

We can write it like so:

```python exec="on" source="material-block" session="dqc"
# Mean-squared error as the comparison criterion
criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Loss function encoding the problem to solve."""
    # Equation loss
    model_output = model(inputs)
    deriv_model = calc_deriv(model_output, inputs)
    deriv_exact = dfdx_equation(inputs)
    ode_loss = criterion(deriv_model, deriv_exact)

    # Boundary loss, f(0) = 0
    boundary_model = model(torch.tensor([[0.0]]))
    boundary_exact = torch.tensor([[0.0]])
    boundary_loss = criterion(boundary_model, boundary_exact)

    return ode_loss + boundary_loss
```

Different loss criterions could be considered, and we could also play with the balance between the sum of the two loss terms. For now, let's proceed with the definition above.

Note that so far we have not used any quantum specific assumption, and we could in principle use the same loss function with a classical neural network.

## Defining a QNN with Qadence

Now, we can finally use Qadence to write a QNN. We will use a feature map to encode the input values, a trainable ansatz circuit, and an observable to measure as the output.

```python exec="on" source="material-block" session="dqc"
from qadence import feature_map, hea, chain
from qadence import QNN, QuantumCircuit, Z
from qadence.types import BasisSet, ReuploadScaling

n_qubits = 3
depth = 3

# Feature map
fm = feature_map(
    n_qubits = n_qubits,
    param = "x",
    fm_type = BasisSet.CHEBYSHEV,
    reupload_scaling = ReuploadScaling.TOWER,
)

# Ansatz
ansatz = hea(n_qubits = n_qubits, depth = depth)

# Observable
observable = Z(0)

circuit = QuantumCircuit(n_qubits, chain(fm, ansatz))
model = QNN(circuit = circuit, observable = observable, inputs = ["x"])
```

We used a Chebyshev feature map with a tower-like scaling of the input reupload, and a standard hardware-efficient ansatz. You can check the [qml constructors tutorial](../../content/qml_constructors.md) to see how you can customize these components. In the observable, for now we consider the simple case of measuring the magnetization of the first qubit.

```python exec="on" source="material-block" html="1" session="dqc"
from qadence.draw import html_string # markdown-exec: hide
from qadence.draw import display

# display(circuit)

print(html_string(circuit)) # markdown-exec: hide
```

## Training the model

Now that the model is defined we can proceed with the training. the `QNN` class can be used like any other `torch.nn.Module`. Here we write a simple training loop, but you can also look at the [ml tools tutorial](ml_tools/trainer.md) to use the convenience training functions that Qadence provides.

To train the model, we will select a random set of collocation points uniformly distributed within $-1.0< x <1.0$ and compute the loss function for those points.

```python exec="on" source="material-block" session="dqc"
n_epochs = 200
n_points = 10

xmin = -0.99
xmax = 0.99

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Training data. We unsqueeze essentially making each batch have a single x value.
    x_train = (xmin + (xmax-xmin)*torch.rand(n_points, requires_grad = True)).unsqueeze(1)

    loss = loss_fn(inputs = x_train, model = model)
    loss.backward()
    optimizer.step()
```

Note the values of $x$ are only picked from $x\in[-0.99, 0.99]$ since we are using a Chebyshev feature map, and derivative of $\text{acos}(x)$ diverges at $-1$ and $1$.

## Plotting the results

```python exec="on" source="material-block" html="1" session="dqc"
import matplotlib.pyplot as plt

def f_exact(x: torch.Tensor) -> torch.Tensor:
    return 5*(x**4 + (1/3)*x**3 - x**2 - 0.5*x)

x_test = torch.arange(xmin, xmax, step = 0.01).unsqueeze(1)

result_exact = f_exact(x_test).flatten()

result_model = model(x_test).flatten().detach()

plt.clf()  # markdown-exec: hide
plt.figure(figsize=(6, 5))  # markdown-exec: hide
plt.plot(x_test, result_exact, label = "Exact solution")
plt.plot(x_test, result_model, label = " Trained model")
plt.xlabel("x")  # markdown-exec: hide
plt.ylabel("f(x)")  # markdown-exec: hide
plt.xlim((-1.1, 1.1))  # markdown-exec: hide
plt.ylim((-3, 2))  # markdown-exec: hide
plt.legend()  # markdown-exec: hide

from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```

Clearly, the result is not optimal.

## Improving the solution

One point to consider when defining the QNN is the possible output range, which is bounded by the spectrum of the chosen observable. For the magnetization of a single qubit, this means that the output is bounded between -1 and 1, which we can clearly see in the plot.

One option would be to define the observable as the total magnetization over all qubits, which would allow a range of -3 to 3.

```python exec="on" source="material-block" session="dqc"
from qadence import add

observable = add(Z(i) for i in range(n_qubits))

model = QNN(circuit = circuit, observable = observable, inputs = ["x"])

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Training data
    x_train = (xmin + (xmax-xmin)*torch.rand(n_points, requires_grad = True)).unsqueeze(1)

    loss = loss_fn(inputs = x_train, model = model)
    loss.backward()
    optimizer.step()
```

And we again plot the result:

```python exec="on" source="material-block" html="1" session="dqc"
x_test = torch.arange(xmin, xmax, step = 0.01).unsqueeze(1)

result_exact = f_exact(x_test).flatten()

result_model = model(x_test).flatten().detach()

plt.clf()  # markdown-exec: hide
plt.figure(figsize=(6, 5))  # markdown-exec: hide
plt.plot(x_test, result_exact, label = "Exact solution")
plt.plot(x_test, result_model, label = "Trained model")
plt.xlabel("x")  # markdown-exec: hide
plt.ylabel("f(x)")  # markdown-exec: hide
plt.xlim((-1.1, 1.1))  # markdown-exec: hide
plt.ylim((-3, 2))  # markdown-exec: hide
plt.legend()  # markdown-exec: hide

from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


## References

[^1]: [Kyriienko et al., Solving nonlinear differential equations with differentiable quantum circuits.](https://arxiv.org/abs/2011.10395)
