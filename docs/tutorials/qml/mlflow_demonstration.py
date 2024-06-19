from __future__ import annotations

import os
from itertools import count

import torch
from torch.utils.data import DataLoader

from qadence import QuantumCircuit, Z, hea
from qadence.ml_tools import TrainConfig, train_with_grad
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN, QuantumModel
from qadence.types import ExperimentTrackingTool

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
os.environ["MLFLOW_EXPERIMENT"] = "mlflow_demonstration"
os.environ["MLFLOW_RUN_NAME"] = "test_0"

hyperparams = {
    "batch_size": 10,
    "n_qubits": 2,
    "ansatz_depth": 1,
    "observable": Z,
}


# in case you want to track remotely
# os.environ['MLFLOW_TRACKING_USERNAME'] =
# s.environ['MLFLOW_TRACKING_PASSWORD'] =
def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


data = dataloader(hyperparams["batch_size"])
model = QNN(
    QuantumCircuit(
        hyperparams["n_qubits"], hea(hyperparams["n_qubits"], hyperparams["ansatz_depth"])
    ),
    observable=hyperparams["observable"](0),
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


config = TrainConfig(
    folder="mlflow_demonstration",
    max_iter=10,
    checkpoint_every=1,
    write_every=1,
    tracking_tool=ExperimentTrackingTool.MLFLOW,
    hyperparams=hyperparams,
)
train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

os.system("mlflow ui --port 5000")
os.system("mlflow ui --backend-store-uri sqlite:///mlflow.db")
