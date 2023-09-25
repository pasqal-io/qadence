from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import torch

from qadence.backend import BackendName
from qadence.backends.pytorch_wrapper import DiffMode

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from qadence import QNN, RX, RY, HamEvo, Parameter, QuantumCircuit, Z, add, chain, hea, kron, tag
from qadence.blocks import AbstractBlock
from qadence.transpile import set_trainable

# functions to fit
sin_fn = lambda x: np.sin(x)  # noqa: E731
x_2 = lambda x: x**2  # noqa: E731


def hardware_efficient_ansatz(n_qubits: int = 2, depth: int = 1) -> AbstractBlock:
    return hea(n_qubits=n_qubits, depth=depth)


def digital_analog_ansatz(
    h_generator: AbstractBlock, n_qubits: int = 2, depth: int = 1, t_evo: float = 1.0
) -> AbstractBlock:
    time_evolution = HamEvo(h_generator, t_evo)

    it = itertools.count()
    ops = []
    for _ in range(depth):
        layer = kron(
            chain(gate(n, f"theta{next(it)}") for gate in [RX, RY, RX]) for n in range(n_qubits)
        )
        ops.append(chain(layer, time_evolution))
    return chain(*ops)


def quantum_circuit(n_qubits: int = 2, depth: int = 1, use_digital_analog: bool = False):
    # Chebyshev feature map with input parameter defined as non trainable
    phi = Parameter("phi", trainable=False)
    fm = chain(RY(i, phi) for i in range(n_qubits))
    tag(fm, "feature_map")

    if not use_digital_analog:
        # hardware-efficient ansatz
        ansatz = hardware_efficient_ansatz(n_qubits=n_qubits, depth=depth)
    else:
        # Hamiltonian evolution ansatz (digital-analog)
        t_evo = 3.0  # length of the time evolution
        h_generator = add(Z(i) for i in range(n_qubits))  # use total magnetization as Hamiltonian
        ansatz = digital_analog_ansatz(h_generator, n_qubits=n_qubits, depth=depth, t_evo=t_evo)

    tag(ansatz, "ansatz")

    # add a final fixed layer or rotations
    fixed_layer = chain(RY(i, np.pi / 2) for i in range(n_qubits))
    tag(fixed_layer, "fixed")

    return QuantumCircuit(n_qubits, fm, ansatz, fixed_layer)


def get_training_data(
    fn: Callable, domain: tuple = (0, 2 * np.pi), n_teacher: int = 100
) -> tuple[torch.tensor, torch.tensor]:
    start, end = domain
    x_rand_np = np.sort(np.random.uniform(low=start, high=end, size=n_teacher))
    y_rand_np = fn(x_rand_np)

    x_rand = torch.tensor(x_rand_np)
    y_rand = torch.tensor(y_rand_np)

    return x_rand, y_rand


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    do_plotting = False
    use_digital_analog = False  # use a digital-analog ansatz
    n_qubits = 4
    depth = 2

    # initialize the training data
    x_train, y_train = get_training_data(sin_fn, n_teacher=30)

    # initialize the quantum circuit
    circuit = quantum_circuit(n_qubits=n_qubits, depth=depth, use_digital_analog=use_digital_analog)

    # select an observable
    # observable = total_magnetization(n_qubits=n_qubits)
    # FIXME: how to get only one trainable parameter here?
    w = Parameter("w")
    observable = add(Z(i) * w for i in range(n_qubits))

    # create the quantum model to use for optimization
    model = QNN(circuit, observable=observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD)

    ###############################
    ####### train the model #######
    ###############################

    # initialize randomly the variational parameters
    init_params = torch.randn(model.num_vparams)
    model.reset_vparams(init_params)

    # train the model
    n_epochs = 50
    lr = 1.0

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Train")
    print(f"Initial loss: {mse_loss(model(x_train), y_train)}")

    x_test, _ = get_training_data(sin_fn)
    y_pred_initial = model(x_test)
    y_pred_initial_np = y_pred_initial.detach().numpy()

    running_loss = 0.0
    for i in range(n_epochs):
        optimizer.zero_grad()

        loss = mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        print(f"Epoch {i+1} training - Loss: {loss.item()}")

    y_pred = model(x_test)

    x_train_np = x_train.detach().numpy().flatten()
    y_train_np = y_train.detach().numpy().flatten()
    x_test_np = x_test.detach().numpy().flatten()
    y_pred_np = y_pred.detach().numpy().flatten()

    if do_plotting:
        plt.figure()
        plt.scatter(x_train, y_train, label="Training points", marker="o", color="orange")
        plt.plot(x_test, y_pred_initial_np, label="Initial prediction", color="green", alpha=0.5)
        plt.plot(x_test, y_pred_np, label="Final prediction")
        plt.legend()
        plt.show()

    ###################################
    ####### extremize the model #######
    ###################################

    print("Extremize")

    # get the optimal model parameters
    optimal_parameters = model.vparams

    # freeze ansatz and make feature map trainable
    set_trainable(circuit.get_blocks_by_tag("feature_map"), value=True)
    set_trainable(circuit.get_blocks_by_tag("ansatz"), value=False)
    set_trainable(observable, value=False)

    # make another QNN for extremization
    extremize_model = QNN(
        circuit, observable=observable, backend=BackendName.PYQTORCH, diff_mode=DiffMode.AD
    )

    # perform extremization
    lr_extr = 0.5
    n_epochs_extr = 20
    optimizer = torch.optim.Adam(extremize_model.parameters(), lr=lr_extr)

    running_loss = 0.0
    for i in range(n_epochs_extr):
        optimizer.zero_grad()

        # find the maximum
        loss = -1.0 * extremize_model(optimal_parameters)
        loss.backward()
        optimizer.step()

        print(f"Epoch {i+1} extremization - Loss: {loss.item()}")

    x_max = extremize_model.vparams["phi"]
    y_max = model(x_max)
    x_max_np = x_max.detach().numpy().flatten()
    y_max_np = y_max.detach().numpy().flatten()

    if do_plotting:
        plt.figure()
        plt.scatter(x_max_np, y_max_np, label="Extram", marker="*", color="orange", sizes=[100])
        plt.plot(x_test, y_pred_np, label="Final prediction")
        plt.plot()
        plt.legend()
        plt.show()
