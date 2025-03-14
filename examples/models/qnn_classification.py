from __future__ import annotations

import random

import mlflow
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from qadence import QNN, RX, FeatureParameter, QuantumCircuit, Z, chain, hea, kron
from qadence.ml_tools import TrainConfig, Trainer


class IrisDataset(Dataset):
    def __init__(self):
        X, y = load_iris(return_X_y=True)

        scaler = StandardScaler()
        self.X = torch.tensor(scaler.fit_transform(X), requires_grad=False)
        self.y = torch.tensor(y, requires_grad=False)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)


def main():
    n_features = 4  # sepal length, sepal width, petal length, petal width
    n_layers = 3
    n_neurons_final_linear_layer = 3
    n_epochs = 1000
    lr = 1e-1

    mlflow.log_params(
        {
            "n_layers_reuploading": n_layers,
            "n_qubits": n_features,
            "n_neurons_final_linear_layer": n_neurons_final_linear_layer,
            "n_epochs": n_epochs,
            "lr": lr,
        }
    )
    dataset = IrisDataset()

    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    feature_parameters = [FeatureParameter(f"x_{i}") for i in range(n_features)]
    fm_layer = RX(0, feature_parameters[0])
    for q in range(1, n_features):
        fm_layer = kron(fm_layer, RX(q, feature_parameters[q]))

    ansatz_layers = [
        hea(n_qubits=n_features, depth=1, param_prefix=f"theta_{layer}")
        for layer in range(n_layers)
    ]
    blocks = chain(fm_layer, ansatz_layers[0])
    for layer in range(1, n_layers):
        blocks = chain(blocks, fm_layer, ansatz_layers[layer])

    qc = QuantumCircuit(n_features, blocks)
    # savefig(qc, "qc.png")
    qnn = QNN(circuit=qc, observable=Z(0), inputs=[f"x_{i}" for i in range(n_features)])

    model = nn.Sequential(qnn, nn.Linear(1, n_neurons_final_linear_layer), nn.Softmax(dim=1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    def cross_entropy(model: nn.Module, data: Tensor) -> tuple[Tensor, dict]:
        x, y = data
        out = model(x)
        loss = criterion(out, y)
        return loss, {}

    train_config = TrainConfig(max_iter=n_epochs, print_every=10, create_subfolder_per_run=True)
    trainer = Trainer(model, opt, train_config, cross_entropy)

    trainer.fit(dataloader)

    # Final accuracy on a random set of points
    idx_test = random.sample(range(len(dataset)), 50)
    X_test, y_test = dataset[idx_test]

    print((torch.argmax(model(X_test), dim=1) == y_test).type(torch.float32).mean().item())
    accuracy_test = (torch.argmax(model(X_test)) == y_test).type(torch.float32).mean()

    mlflow.log_metric("Test Accuracy", accuracy_test.item())

    mlflow.pytorch.log_model(
        model, artifact_path="final_model", input_example=X_test.detach().numpy()
    )


if __name__ == "__main__":
    main()
