In this tutorial we will show how to use Qadence to solve a basic classification task using a hybrid quantum-classical model composed of a QNN and classical layers.

## Dataset

We will use the Iris dataset separated into training and testing sets. A StandardScaler is used to normalize the data.

```python exec="on" source="material-block" session="classification"

import random

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from qadence import QNN, RX, FeatureParameter, QuantumCircuit, Z, chain, hea, kron
from qadence.ml_tools import TrainConfig, Trainer

class IrisDataset(Dataset):
    """The Iris dataset split into a training set and a test set.

    A StandardScaler is applied prior to applying models.
    """

    def __init__(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.X = torch.tensor(self.scaler.transform(X_train), requires_grad=False)
        self.y = torch.tensor(y_train, requires_grad=False)

        self.X_test = torch.tensor(self.scaler.transform(X_test), requires_grad=False)
        self.y_test = torch.tensor(y_test, requires_grad=False)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)

n_features = 4  # sepal length, sepal width, petal length, petal width
n_layers = 3
n_neurons_final_linear_layer = 3
n_epochs = 1000
lr = 1e-1
dataset = IrisDataset()

dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

```


## Hybrid QNN

We set up the QNN part composed of multiple feature map layers, each followed by a variational layer.
The output will be the expectation value with respect to a $Z$ observable on qubit 0
Then we add a simple linear layer serving as as classification head. Note softmax is not applied with the cross-entropy loss.

```python exec="on" source="material-block" session="classification"

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
qnn = QNN(circuit=qc, observable=Z(0), inputs=[f"x_{i}" for i in range(n_features)])
model = nn.Sequential(qnn, nn.Linear(1, n_neurons_final_linear_layer))

```

## Training

Then we can set up the training part:

```python exec="on" source="material-block" session="classification"
opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def cross_entropy(model: nn.Module, data: Tensor) -> tuple[Tensor, dict]:
    x, y = data
    out = model(x)
    loss = criterion(out, y)
    return loss, {}

train_config = TrainConfig(max_iter=n_epochs, print_every=10, create_subfolder_per_run=True)
Trainer.set_use_grad(True)
trainer = Trainer(model=model, optimizer=opt, config=train_config, loss_fn=cross_entropy)


res_train = trainer.fit(dataloader)
```

## Inference

Finally, we can apply our model on the test set and check the score.

```python exec="on" source="material-block" session="classification"
X_test, y_test = dataset.X_test, dataset.y_test
preds_test = torch.argmax(torch.softmax(model(X_test), dim=1), dim=1)
accuracy_test = (preds_test == y_test).type(torch.float32).mean()
## Should reach higher than 0.9
print("Test Accuracy", accuracy_test.item()) # markdown-exec: hide
```
