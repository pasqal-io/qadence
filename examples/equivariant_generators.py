"""This example implements some random equivariant Hamiltonians and
checks that the energy obtained by quantum circuit expectation value
coincides with the one computed with exact diagonalization

Use this example to see how to leverage openfermion for building
complex observables
"""
from __future__ import annotations

import numpy as np
import torch
from openfermion import QubitOperator, get_sparse_operator, hermitian_conjugated

from qadence import QNN, RY, Parameter, QuantumCircuit, chain, hea, tag
from qadence.backend import BackendName
from qadence.backends.pytorch_wrapper import DiffMode
from qadence.blocks import AbstractBlock
from qadence.blocks.manipulate import from_openfermion

np.random.seed(42)
torch.manual_seed(42)


# write a commutator for 2 operators
def commutator(op1, op2):
    return op1 * op2 - op2 * op1


def twirling_operator_openfermion(pauli_op, list_generators, list_generators_adjoint):
    """Compute the twirling operator of a given Pauli operator.
    Args:
        pauli_op (Pauli): Pauli operator
        list_generators (list): List of generators
        list_generators_adjoint (list): List of adjoint generators
    Returns:
        Operator: Twirling operator.
    """
    twirling_op = 0
    for generator, generator_adjoint in zip(list_generators, list_generators_adjoint):
        twirling_op += generator * pauli_op * generator_adjoint
    return 1 / len(list_generators) * twirling_op


def pauli_combinations():
    pauli_list = ["X", "Y", "Z", "I"]
    # write all the possible combinations of the pauli_list
    combs = []
    for i in range(len(pauli_list)):
        for j in range(len(pauli_list)):
            if pauli_list[i] != "I" and pauli_list[j] != "I":
                combs.append(str(pauli_list[i] + "0 " + pauli_list[j] + "1"))
            elif pauli_list[i] != "I" and pauli_list[j] == "I":
                combs.append(str(pauli_list[i] + "0"))
            elif pauli_list[i] == "I" and pauli_list[j] != "I":
                combs.append(str(pauli_list[j] + "1"))
            else:
                pass
    return combs


def generate_random_hamiltonian(list_twiling_operators):
    hamiltonian = 0.0  # * QubitOperator("") # identity
    # generate random coefficients from uniform distribution [0,1)
    # random_coefficients = np.random.random((len(list_twiling_operators[:-1]),))
    size = len(list_twiling_operators[:-1])
    random_coefficients = np.random.uniform(-1, 1, size)

    for idx, op in enumerate(list_twiling_operators[:-1]):
        hamiltonian += random_coefficients[idx] * op

    # Diagonalize the Hamiltonian
    h_matrix = get_sparse_operator(hamiltonian, n_qubits=None, trunc=None, hbar=1.0).todense()

    eigvals, eigvecs = np.linalg.eig(h_matrix)
    # sort eigenvalues and eigenvectors
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs.T
    eigvecs = eigvecs[idx]

    return [hamiltonian, eigvals[0], eigvecs[0]]


def quantum_circuit(n_qubits: int = 2, depth: int = 1, use_digital_analog: bool = False):
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(*[RY(i, phi) for i in range(n_qubits)])
    tag(fm, "feature_map")

    ansatz = hea(n_qubits=n_qubits, depth=depth)
    tag(ansatz, "ansatz")

    return QuantumCircuit(n_qubits, fm, ansatz)


n_hamiltonians = 5

list_hamiltonians = []
list_exact_energies = []
list_eigenvecs = []
list_twiling_operators = []

#
# Analytical test with Openfermion operators
#

# generators for the equivariant group
g0 = QubitOperator("")
g1 = 1 / 2 * (
    QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
) + 1 / 2 * QubitOperator(
    ""
)  # latter is the identity
list_generators = [g0, g1]

# create the adjoint of the generators
list_generators_adjoint = []
for qubit_operator in list_generators:
    list_generators_adjoint.append(hermitian_conjugated(qubit_operator))

combs = pauli_combinations()
for pauli in combs:
    top = twirling_operator_openfermion(
        QubitOperator(pauli), list_generators, list_generators_adjoint
    )
    list_twiling_operators.append(top)

# write a commutator for 2 operators
for i in range(n_hamiltonians):
    h, e, ev = generate_random_hamiltonian(list_twiling_operators)

    list_hamiltonians.append(h)
    list_exact_energies.append(e)
    list_eigenvecs.append(ev)

    # Check that ham commutes with all the generators
    for generator in list_generators:
        assert commutator(h, generator).__repr__() == "0"

# Retrieve the ground state energy with quantum circuit


def train_equivariant(observable: AbstractBlock, expected_energy: float):
    print(f"Selected observable: {observable}")

    # define the quantum model
    circuit = quantum_circuit(n_qubits=2, depth=2)
    model = QNN(circuit, observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

    # initialize randomly the variational parameters
    init_params = torch.rand(model.num_vparams)
    model.reset_vparams(init_params)

    # train the model
    n_epochs = 500
    lr = 5e-2

    batch_size = 1
    input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
    y_train = -10 * torch.ones_like(torch.rand(batch_size, requires_grad=True))

    mse_loss = torch.nn.MSELoss()  # standard PyTorch loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # standard PyTorch Adam optimizer
    pred = model(input_values)
    print(f"Initial loss: {mse_loss(pred, y_train)}")

    for i in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(input_values)
        loss = mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch {i+1} - Loss: {loss.item()} - Energy: {pred.item()}")

    assert np.isclose(float(model(input_values)), expected_energy, atol=1e-4, rtol=1e-4)


# select one observable
for ham, energy in zip(list_hamiltonians, list_exact_energies):
    train_equivariant(from_openfermion(ham), energy)
