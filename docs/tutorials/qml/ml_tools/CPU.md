# Training on CPU with `Trainer`

This guide explains how to train models on **CPU** using `Trainer` from `qadence.ml_tools`, covering **single-process** and **multi-processing** setups.

### Understanding Arguments
- *spawn*: If True, enables multi-processing and launches separate processes.
- *nprocs*: Number of processes to run.
- *compute_setup*: The computational setup used for training. Options include `cpu`, `gpu`, and `auto`.

For more details on the advanced training options, please refer to [TrainConfig Documentation](./data_and_config.md)

## **Configuring `TrainConfig` for CPU Training**

By adjusting `TrainConfig`, you can seamlessly switch between single and multi-core CPU training. To enable CPU-based training, update these fields in `TrainConfig`:

### Single-Process Training Configuration:
- **`backend="cpu"`**: Ensures training runs on the CPU.
- **`spawn=False`**: Uses a single process (default).
- **`nprocs=1`**: Uses one CPU core.

```python
train_config = TrainConfig(
    compute_setup="cpu",
)
```

### Multi-Processing Configuration
- **`backend="gloo"`**: Uses the Gloo backend for CPU multi-processing.
- **`spawn=True`**: Enables multi-processing.
- **`nprocs=4`**: Utilizes 4 CPU cores.

```python
train_config = TrainConfig(
    compute_setup="cpu",
    backend="gloo",
    spawn=True,
    nprocs=4,
)
```

## Examples

### Single-Process CPU Training Example

Single-Process Training: Simple and suitable for small datasets. Use `backend="cpu"`, `spawn=False`.

```python exec="on" source="material-block" result="json"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import TrainConfig, Trainer
from qadence.ml_tools.optimize_step import optimize_step
Trainer.set_use_grad(True)

# Dataset, Model, and Optimizer
x = torch.linspace(0, 1, 100).reshape(-1, 1)
y = torch.sin(2 * torch.pi * x)
dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Single-Process Training Configuration
train_config = TrainConfig(compute_setup="cpu", max_iter=5, print_every=1)

# Training
trainer = Trainer(model, optimizer, train_config, loss_fn="mse", optimize_step=optimize_step)
trainer.fit(dataloader)
```


### Multi-Processing CPU Training Example

Multi-Processing Training: Best for large datasets, utilizes multiple CPU processes. Use `backend="gloo"`, `spawn=True`, and set `nprocs`.

```python exec="on" source="material-block" result="json" html="1"
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import TrainConfig, Trainer
from qadence.ml_tools.optimize_step import optimize_step
Trainer.set_use_grad(True)

# __main__ is recommended.
if __name__ == "__main__":
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Multi-Process Training Configuration
    train_config = TrainConfig(
        compute_setup="cpu",
        backend="gloo",
        spawn=True,
        nprocs=4,
        max_iter=5,
        print_every=1)

    trainer = Trainer(model, optimizer, train_config, loss_fn="mse", optimize_step=optimize_step)
    trainer.fit(dataloader)
```
