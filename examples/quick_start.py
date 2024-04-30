from __future__ import annotations

import torch

from qadence import QuantumModel  # quantum model for execution

# qadence has many submodules
from qadence.blocks import kron  # block system
from qadence.circuit import QuantumCircuit  # circuit to assemble quantum operations
from qadence.ml_tools import TrainConfig, train_with_grad  # tools for ML simulations
from qadence.operations import RX, HamEvo, X, Y, Zero  # quantum operations
from qadence.parameters import VariationalParameter  # trainable parameters

# all of the above can also be imported directly from the qadence namespace

n_qubits = 4
n_circ_params = n_qubits

# define some variational parameters
circ_params = [VariationalParameter(f"theta{i}") for i in range(n_circ_params)]

# block with single qubit rotations
rot_block = kron(RX(i, param) for i, param in enumerate(circ_params))

# block with Hamiltonian evolution
t_evo = 2.0
generator = 0.25 * X(0) + 0.25 * X(1) + 0.5 * Y(2) + 0.5 * Y(3)
ent_block = HamEvo(generator, t_evo)

# create an observable to measure with tunable coefficients
obs_params = [VariationalParameter(f"phi{i}") for i in range(n_qubits)]
obs = Zero()
for i in range(n_qubits):
    obs += obs_params[i] * X(i)

# create circuit and executable quantum model
circuit = QuantumCircuit(n_qubits, rot_block, ent_block)
model = QuantumModel(circuit, observable=obs, diff_mode="ad")

samples = model.sample({}, n_shots=1000)
print(samples)  # this returns a Counter instance

# compute the expectation value of the observable
expval = model.expectation({})
print(expval)


# define a loss function and train the model
# using qadence built-in ML tools
def loss_fn(model_: QuantumModel, _):
    return model_.expectation({}).squeeze(), {}


optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
config = TrainConfig(max_iter=100, checkpoint_every=10, print_every=10)
train_with_grad(model, None, optimizer, config, loss_fn=loss_fn)
