from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
from torch.autograd import grad

from qadence import QNN, BasisSet, QuantumCircuit, feature_map, hea, total_magnetization

torch.manual_seed(42)
np.random.seed(42)

do_plotting = False


# Equation we want to learn
def f(x):
    return 3 * x**2 + 2 * x - 1


# calculation of constant terms
# -> d2ydx2 = 6
# -> dydx   = 6 * x + 2
# dy[0] = 2
# y[0]  = -1


# Using torch derivatives directly
def d2y(ufa, x):
    y = ufa(x)
    dydx = grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    d2ydx2 = grad(dydx, x, torch.ones_like(dydx), create_graph=True, retain_graph=True)[0]
    return d2ydx2 - 6.0


def dy0(ufa, x):
    y = ufa(x)
    dydx = grad(y, x, torch.ones_like(y), create_graph=True)[0]
    return dydx - 2.0


n_qubits = 5
batch_size = 100
x = torch.linspace(-0.5, 0.5, batch_size).reshape(batch_size, 1).requires_grad_()
x0 = torch.zeros((1, 1), requires_grad=True)
x1 = torch.zeros((1, 1), requires_grad=True)

fm = feature_map(n_qubits=5, fm_type=BasisSet.CHEBYSHEV)
ansatz = hea(n_qubits=5, depth=5, periodic=True)
circ = QuantumCircuit(5, fm, ansatz)
ufa = QNN(circ, observable=total_magnetization(n_qubits=5))

x = torch.linspace(-0.5, 0.5, 100).reshape(-1, 1)
y = ufa(x)

if do_plotting:
    xn = x.detach().numpy().reshape(-1)
    yn = y.detach().numpy().reshape(-1)
    yt = f(x)
    plt.plot(xn, yt, label="Truth")
    plt.plot(xn, yn, label="Pred.")
    plt.legend()
    plt.show()
