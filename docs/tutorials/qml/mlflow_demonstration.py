from __future__ import annotations

import os
import random
from itertools import count

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from qadence import QuantumCircuit, Z, hea
from qadence.constructors import feature_map, hamiltonian_factory
from qadence.ml_tools import TrainConfig, train_with_grad
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN, QuantumModel
from qadence.types import ExperimentTrackingTool

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_EXPERIMENT"] = "mlflow_demonstration"
os.environ["MLFLOW_RUN_NAME"] = "test_0"

hyperparams = {
    "seed": 42,
    "batch_size": 10,
    "n_qubits": 2,
    "ansatz_depth": 1,
    "observable": Z,
}

np.random.seed(hyperparams["seed"])
torch.manual_seed(hyperparams["seed"])
random.seed(hyperparams["seed"])


# in case you want to track remotely
# os.environ['MLFLOW_TRACKING_USERNAME'] =
# s.environ['MLFLOW_TRACKING_PASSWORD'] =
def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


obs = hamiltonian_factory(register=hyperparams["n_qubits"], detuning=hyperparams["observable"])

data = dataloader(hyperparams["batch_size"])
fm = feature_map(hyperparams["n_qubits"], param="x")
model = QNN(
    QuantumCircuit(
        hyperparams["n_qubits"], fm, hea(hyperparams["n_qubits"], hyperparams["ansatz_depth"])
    ),
    observable=obs,
    inputs=["x"],
)
cnt = count()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
inputs = rand_featureparameters(model, 1)


def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    out = model.expectation(inputs)
    loss = criterion(out, torch.rand(1))
    return loss, {}


def plot_fn(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
    descr = f"ufa_prediction_epoch_{iteration}.png"
    fig, ax = plt.subplots()
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    out = model.expectation(x)
    ax.plot(x.detach().numpy(), out.detach().numpy())
    return descr, fig


config = TrainConfig(
    folder="mlflow_demonstration",
    max_iter=10,
    checkpoint_every=1,
    plot_every=2,
    write_every=1,
    tracking_tool=ExperimentTrackingTool.MLFLOW,
    hyperparams=hyperparams,
    plotting_functions=(plot_fn,),
)
train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

os.system("mlflow ui --port 5000")
os.system("mlflow ui --backend-store-uri sqlite:///mlflow.db")
