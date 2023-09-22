from __future__ import annotations

import numpy as np
import torch

from qadence import hea, tag
from qadence.draw import savefig
from qadence.utils import samples_to_integers

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


from dqgm_model import DQGM

"""
This example script showcases the DQGM algorithm from the paper "Protocols for Trainable
and Differentiable Quantum Generative Modelling".

The DQGM algorithm is further explored in the QGenMod library.

"""


def normalpdf(x: np.ndarray, sigma=1.0, mu=0.0):
    """
    Example distribution we are going to approximate

    Args:
        x: stochastic variable
        sigma: standard deviation
        mu: mean
    Returns:
        normal PDF
    """
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / sigma / np.sqrt(2 * np.pi)


if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ###########
    ## INPUT ##
    ###########

    # Stochastic variable range
    xmin = -1.0
    xmax = 2.0
    delta_x = xmax - xmin

    n_points = 100
    x_train = torch.unsqueeze(torch.linspace(xmin, xmax, n_points), 1)

    # Setting up the target PDF
    x_train_np = x_train.detach().numpy().flatten()

    # Double Gaussian:
    y_train = torch.tensor(
        (
            normalpdf(x_train_np, sigma=delta_x / 20, mu=0)
            + normalpdf(x_train_np, sigma=delta_x / 10, mu=xmax - 1)
        )
        / 2
    )

    # Qubits for the ansatz:
    # Determines/limits the expressibility/trainability
    n_qubits_training = 3

    # Qubit resolution for sampling, and # of samples:
    n_qubits_resolution = 6
    n_samples = 5000

    visualize = False  # whether to plot figures and store

    ###########
    ## MODEL ##
    ###########

    ansatz = hea(n_qubits=n_qubits_training, depth=2)

    tag(ansatz, "ansatz")

    model = DQGM(
        ansatz,
        n_features=1,
        n_qubits_per_feature=n_qubits_training,
        feature_range=(xmin, xmax),
    )

    if visualize:
        savefig(model.circuit, "dqgm_qnn_circuit.png")

    ##############
    ## TRAINING ##
    ##############

    out_initial_qnn = model(x_train)

    # set up hyper-parameters and train the model against the target PDF
    n_epochs = 100
    lr = 0.25
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting Training the DQGM")
    print(
        f"Initial loss: {mse_loss(out_initial_qnn * (2**model.n_qubits_total) / delta_x, y_train)}"
    )

    ls = []
    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = mse_loss(model(x_train) * (2**model.n_qubits_total) / (delta_x), y_train)
        loss.backward()
        optimizer.step()

        ls.append(loss.item())
        print(f"Epoch {i+1} training - Loss: {loss.item()}")

    print("Finished training.")

    out_final_qnn = model(x_train)

    ######################
    ## SAMPLING RESULTS ##
    ######################

    print("Setting up corresponding generator model")
    model.init_sampling_model(n_qubits_resolution)
    if visualize:
        savefig(model._sampling_model.backend.abstract_circuit, "dqgm_sampling_circuit.png")

    print("Compute the generator output probabilities exactly with WF simulator")
    p = ((2**n_qubits_resolution) / delta_x) * model.probabilities().detach().numpy().flatten()

    print("Sampling using the trained ansatz...")
    samples = model.sample(n_samples)

    ###################
    ## VISUALIZATION ##
    ###################

    if visualize:
        # setting up the plots
        fig, ax = plt.subplots(2, figsize=(6, 8), gridspec_kw={"height_ratios": [1, 2]})

        # plotting the loss
        ax[0].semilogy(ls, color="r", label="Training loss")
        ax[0].set_xlabel("Training epoch #")
        ax[0].set_ylabel("Loss MSE(QNN(x), pdf(x))")
        ax[0].legend(loc="upper right")
        ax[0].set_title(f"DQGM: {n_qubits_resolution}-q resolution & {n_qubits_training}-q ansatz")

        # plotting the target PDF
        ynp = y_train.detach().numpy().flatten()
        ax[1].plot(x_train_np, ynp, color="red", label="target PDF")

        # histogram
        int_samples = samples_to_integers(samples)

        int_vals = np.array(list(int_samples.keys()))
        counts = np.array(list(int_samples.values()))

        counts_norm = ((2**n_qubits_resolution) / delta_x) * counts / n_samples
        samples_rescale = (int_vals) * delta_x / (2**n_qubits_resolution) + xmin

        # setting up the histogram bin locations
        xp = np.linspace(xmin, xmax, 2**n_qubits_resolution)

        # plotting the exact sampling probabilities
        ax[1].plot(xp, p, color="lightblue", label="Trained Sampling Probabilities")
        # plotting the samples drawn from the generator circuit
        ax[1].bar(
            samples_rescale,
            counts_norm,
            width=delta_x / (len(xp) - 1),
            color="blue",
            label="Trained Sampling Histogram",
        )

        # plotting the initial QNN model output
        y_initial = (
            out_initial_qnn.detach().numpy().flatten() * (2**model.n_qubits_total) / (delta_x)
        )
        ax[1].plot(x_train_np, y_initial, linestyle="--", color="k", label="initial QNN(x)")

        # plotting the trained QNN model output
        ypred = out_final_qnn.detach().numpy().flatten() * (2**model.n_qubits_total) / (delta_x)
        ax[1].plot(x_train_np, ypred, color="green", label="trained QNN(x)")

        # finishing up
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("pdf(x)")
        plt.show()
