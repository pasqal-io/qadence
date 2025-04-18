
# Qadence Trainer Guide

The [`Trainer`][qadence.ml_tools.Trainer] class in `qadence.ml_tools` is a versatile tool designed to streamline the training of quantum machine learning models.
It offers flexibility for both gradient-based and gradient-free optimization methods, supports custom loss functions, and integrates seamlessly with tracking tools like TensorBoard and MLflow.
Additionally, it provides hooks for implementing custom behaviors during the training process.

For training QML models, Qadence offers this out-of-the-box [`Trainer`][qadence.ml_tools.Trainer] for optimizing differentiable
models, _e.g._ `QNN`s and `QuantumModel`, containing either *trainable* and/or *non-trainable* parameters
(see [the parameters tutorial](../../../content/parameters.md) for detailed information about parameter types):

---

## 1. Overview

The `Trainer` class simplifies the training workflow by managing the training loop, handling data loading, and facilitating model evaluation.
It is compatible with various optimization strategies and allows for extensive customization to meet specific training requirements.

Example of initializing the `Trainer`:

```python
from qadence.ml_tools import Trainer, TrainConfig
from torch.optim import Adam

# Initialize model and optimizer
model = ...  # Define or load a quantum model here
optimizer = Adam(model.parameters(), lr=0.01)
config = TrainConfig(max_iter=100, print_every=10)

# Initialize Trainer with model, optimizer, and configuration
trainer = Trainer(model=model, optimizer=optimizer, config=config)
```

> Notes:
> `qadence` versions prior to 1.9.0 provided `train_with_grad` and `train_no_grad` functions, which are being replaced with `Trainer`. The user can transition as following.
> ```python
> from qadence.ml_tools import train_with_grad
> train_with_grad(model=model, optimizer=optimizer, config=config, data = data)
> ```
> to
> ```python
> from qadence.ml_tools import Trainer
> trainer = Trainer(model=model, optimizer=optimizer, config=config)
> trainer.fit(train_dataloader = data)
> ```

## 2. Gradient-Based and Gradient-Free Optimization

The `Trainer` supports both gradient-based and gradient-free optimization methods.
Default is gradient-based optimization.

- **Gradient-Based Optimization**: Utilizes optimizers from PyTorch's `torch.optim` module.
This is the default behaviour of the `Trainer`, thus setting this is not necessary.
However, it can be explicity mentioned as follows.
Example of using gradient-based optimization:

```python exec="on" source="material-block"
from qadence.ml_tools import Trainer

# set_use_grad(True) to enable gradient based training. This is the default behaviour of Trainer.
Trainer.set_use_grad(True)
```

- **Gradient-Free Optimization**: Employs optimization algorithms from the [Nevergrad](https://facebookresearch.github.io/nevergrad/) library.


Example of using gradient-free optimization with Nevergrad:

```python exec="on" source="material-block"
from qadence.ml_tools import Trainer

# set_use_grad(False) to disable gradient based training.
Trainer.set_use_grad(False)
```

### Using Context Managers for Mixed Optimization

For cases requiring both optimization methods in a single training session, the `Trainer` class provides context managers to enable or disable gradients.

```python
# Temporarily switch to gradient-based optimization
with trainer.enable_grad_opt(optimizer):
    print("Gradient Based Optimization")
    # trainer.fit(train_loader)

# Switch to gradient-free optimization for specific steps
with trainer.disable_grad_opt(ng_optimizer):
    print("Gradient Free Optimization")
    # trainer.fit(train_loader)
```

---

## 3. Custom Loss Functions

Users can define custom loss functions tailored to their specific tasks.
The `Trainer` accepts a `loss_fn` parameter, which should be a callable that takes the model and data as inputs and returns a tuple containing the loss tensor and a dictionary of metrics.

Example of using a custom loss function:

```python exec="on" source="material-block"
import torch
from itertools import count
cnt = count()
criterion = torch.nn.MSELoss()

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data
    out = model(x)
    loss = criterion(out, y)
    return loss, {}
```

This custom loss function can be used in the trainer
```python
from qadence.ml_tools import Trainer, TrainConfig
from torch.optim import Adam

# Initialize model and optimizer
model = ...  # Define or load a quantum model here
optimizer = Adam(model.parameters(), lr=0.01)
config = TrainConfig(max_iter=100, print_every=10)

trainer = Trainer(model=model, optimizer=optimizer, config=config, loss_fn=loss_fn)
```


---

## 4. Hooks for Custom Behavior

The `Trainer` class provides several hooks that enable users to inject custom behavior at different stages of the training process.
These hooks are methods that can be overridden in a subclass to execute custom code.
The available hooks include:

- `on_train_start`: Called at the beginning of the training process.
- `on_train_end`: Called at the end of the training process.
- `on_train_epoch_start`: Called at the start of each training epoch.
- `on_train_epoch_end`: Called at the end of each training epoch.
- `on_train_batch_start`: Called at the start of each training batch.
- `on_train_batch_end`: Called at the end of each training batch.

Each "start" and "end" hook receives data and loss metrics as arguments. The specific values provided for these arguments depend on the training stage associated with the hook. The context of the training stage (e.g., training, validation, or testing) determines which metrics are relevant and how they are populated. For details of inputs on each hook, please review the documentation of [`BaseTrainer`][qadence.ml_tools.train_utils.BaseTrainer].

    - Example of what inputs are provided to training hooks.

        ```
        def on_train_batch_start(self, batch: Tuple[torch.Tensor, ...] | None) -> None:
            """
            Called at the start of each training batch.

            Args:
                batch: A batch of data from the DataLoader. Typically a tuple containing
                    input tensors and corresponding target tensors.
            """
            pass
        ```
        ```
        def on_train_batch_end(self, train_batch_loss_metrics: Tuple[torch.Tensor, Any]) -> None:
            """
            Called at the end of each training batch.

            Args:
                train_batch_loss_metrics: Metrics for the training batch loss.
                    Tuple of (loss, metrics)
            """
            pass
        ```

Example of using a hook to log a message at the end of each epoch:

```python exec="on" source="material-block"
from qadence.ml_tools import Trainer

class CustomTrainer(Trainer):
    def on_train_epoch_end(self, train_epoch_loss_metrics):
        print(f"End of epoch - Loss and Metrics: {train_epoch_loss_metrics}")
```

> Notes:
> Trainer offers inbuilt callbacks as well. Callbacks are mainly for logging/tracking purposes, but the above mentioned hooks are generic. The workflow for every train batch looks like:
> 1. perform on_train_batch_start callbacks,
> 2. call the on_train_batch_start hook,
> 3. do the batch training,
> 4. call the on_train_batch_end hook, and
> 5. perform on_train_batch_end callbacks.
>
> The use of `on_`*{phase}*`_start` and `on_`*{phase}*`_end` hooks is not specifically to add extra callbacks, but for any other generic pre/post processing. For example, reshaping input batch in case of RNNs/LSTMs, post processing loss and adding an extra metric. They could also be used to add more callbacks (which is not recommended - as we provide methods to add extra callbacks in the TrainCofig)

---

## 5. Experiment Tracking with TensorBoard and MLflow

The `Trainer` integrates with TensorBoard and MLflow for experiment tracking:

- **TensorBoard**: Logs metrics and visualizations during training, allowing users to monitor the training process.

- **MLflow**: Tracks experiments, logs parameters, metrics, and artifacts, and provides a user-friendly interface for comparing different runs.

To utilize these tracking tools, the `Trainer` can be configured with appropriate writers that handle the logging of metrics and other relevant information during training.

Example of using TensorBoard tracking:

```python
from qadence.ml_tools import TrainConfig
from qadence.types import ExperimentTrackingTool

# Set up tracking with TensorBoard
config = TrainConfig(max_iter=100, tracking_tool=ExperimentTrackingTool.TENSORBOARD)
```

Example of using MLflow tracking:

```python
from qadence.types import ExperimentTrackingTool

# Set up tracking with MLflow
config = TrainConfig(max_iter=100, tracking_tool=ExperimentTrackingTool.MLFLOW)
```

## 6. Examples

### 6.1. Training with `Trainer` and `TrainConfig`

#### Setup
Let's do the necessary imports and declare a `DataLoader`. We can already define some hyperparameters here, including the seed for random number generators. mlflow can log hyperparameters with arbitrary types, for example the observable that we want to monitor (`Z` in this case, which has a `qadence.Operation` type).

```python
import random
from itertools import count

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.nn import Module
from torch.utils.data import DataLoader

from qadence import hea, QuantumCircuit, Z
from qadence.constructors import feature_map, hamiltonian_factory
from qadence.ml_tools import Trainer, TrainConfig
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.utils import rand_featureparameters
from qadence.ml_tools.models import QNN, QuantumModel
from qadence.types import ExperimentTrackingTool

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


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)
```

We continue with the regular QNN definition, together with the loss function and optimizer.

```python
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
```

#### `TrainConfig` specifications
Qadence offers different tracking options via `TrainConfig`. Here we use the `ExperimentTrackingTool` type to specify that we want to track the experiment with mlflow. Tracking with tensorboard is also possible. We can then indicate *what* and *how often* we want to track or log.

**For Training**
`write_every` controls the number of epochs after which the loss values is logged. Thanks to the `plotting_functions` and `plot_every`arguments, we are also able to plot model-related quantities throughout training. Notice that arbitrary plotting functions can be passed, as long as the signature is the same as `plot_fn` below. Finally, the trained model can be logged by setting `log_model=True`. Here is an example of plotting function and training configuration

```python
def plot_fn(model: Module, iteration: int) -> tuple[str, Figure]:
    descr = f"ufa_prediction_epoch_{iteration}.png"
    fig, ax = plt.subplots()
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    out = model.expectation(x)
    ax.plot(x.detach().numpy(), out.detach().numpy())
    return descr, fig


config = TrainConfig(
    root_folder="mlflow_demonstration",
    max_iter=10,
    checkpoint_every=1,
    plot_every=2,
    write_every=1,
    log_model=True,
    tracking_tool=ExperimentTrackingTool.MLFLOW,
    hyperparams=hyperparams,
    plotting_functions=(plot_fn,),
)
```

#### Training and inspecting
Model training happens as usual
```python
trainer = Trainer(model, optimizer, config, loss_fn)
trainer.fit(train_dataloader=data)
```

After training , we can inspect our experiment via the mlflow UI
```bash
mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```
In this case, since we're running on a local server, we can access the mlflow UI by navigating to http://localhost:8080/.


### 6.2. Fitting a function with a QNN using `ml_tools`

In Quantum Machine Learning, the general consensus is to use `complex128` precision for states and operators and `float64` precision for parameters. This is also the convention which is used in `qadence`.
However, for specific usecases, lower precision can greatly speed up training and reduce memory consumption. When using the `pyqtorch` backend, `qadence` offers the option to move a `QuantumModel` instance to a specific precision using the torch `to` syntax.

Let's look at a complete example of how to use `Trainer` now. Here we perform a validation check during training and use a validation criterion that checks whether the validation loss in the current iteration has decreased compared to the lowest validation loss from all previous iterations. For demonstration, the train and the validation data are kept the same here. However, it is beneficial and encouraged to keep them distinct in practice to understand model's generalization capabilities.

```python exec="on" source="material-block" html="1"
from pathlib import Path
import torch
from functools import reduce
from operator import add
from itertools import count
import matplotlib.pyplot as plt

from qadence import Parameter, QuantumCircuit, Z
from qadence import hamiltonian_factory, hea, feature_map, chain
from qadence import QNN
from qadence.ml_tools import  TrainConfig, Trainer, to_dataloader

Trainer.set_use_grad(True)

n_qubits = 4
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning=Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 100
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

cnt = count()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def loss_fn(model: torch.nn.Module, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
    next(cnt)
    x, y = data[0], data[1]
    out = model(x)
    loss = criterion(out, y)
    return loss, {}

def validation_criterion(
    current_validation_loss: float, current_best_validation_loss: float, val_epsilon: float
) -> bool:
    return current_validation_loss <= current_best_validation_loss - val_epsilon

n_epochs = 300

config = TrainConfig(
    max_iter=n_epochs,
    batch_size=batch_size,
    checkpoint_best_only=True,
    val_every=10,  # The model will be run on the validation data after every `val_every` epochs.
    validation_criterion=validation_criterion
)

fn = lambda x, degree: .05 * reduce(add, (torch.cos(i*x) + torch.sin(i*x) for i in range(degree)), 0.)
x = torch.linspace(0, 10, batch_size).reshape(-1, 1)
y = fn(x, 5)

train_dataloader = to_dataloader(x, y, batch_size=batch_size, infinite=True)
val_dataloader =  to_dataloader(x, y, batch_size=batch_size, infinite=True)

trainer = Trainer(model, optimizer, config, loss_fn=loss_fn,
                    train_dataloader=train_dataloader, val_dataloader=val_dataloader)
trainer.fit()

plt.clf()
plt.plot(x.numpy(), y.numpy(), label='truth')
plt.plot(x.numpy(), model(x).detach().numpy(), "--", label="final", linewidth=3)
plt.legend()
from docs import docsutils # markdown-exec: hide
print(docsutils.fig_to_html(plt.gcf())) # markdown-exec: hide
```


### 6.3. Fitting a function - Low-level API

For users who want to use the low-level API of `qadence`, here an example written without `Trainer`.

```python exec="on" source="material-block"
from pathlib import Path
import torch
from itertools import count
from qadence.constructors import hamiltonian_factory, hea, feature_map
from qadence import chain, Parameter, QuantumCircuit, Z
from qadence import QNN
from qadence.ml_tools import TrainConfig

n_qubits = 2
fm = feature_map(n_qubits)
ansatz = hea(n_qubits=n_qubits, depth=3)
observable = hamiltonian_factory(n_qubits, detuning=Z)
circuit = QuantumCircuit(n_qubits, fm, ansatz)

model = QNN(circuit, observable, backend="pyqtorch", diff_mode="ad")
batch_size = 1
input_values = {"phi": torch.rand(batch_size, requires_grad=True)}
pred = model(input_values)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
n_epochs=50
cnt = count()

tmp_path = Path("/tmp")

config = TrainConfig(
    root_folder=tmp_path,
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
)

x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
y = torch.sin(x)

for i in range(n_epochs):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```



### 6.4. Performing pre-training Exploratory Landscape Analysis (ELA) with Information Content (IC)

Before one embarks on training a model, one may wish to analyze the loss landscape to judge the trainability and catch vanishing gradient issues early.
One way of doing this is made possible via calculating the [Information Content of the loss landscape](https://www.nature.com/articles/s41534-024-00819-8).
This is done by discretizing the gradient in the loss landscapes and then calculating the information content therein.
This serves as a measure of flatness or ruggedness of the loss landscape.
Quantitatively, the information content allows us to get bounds on the average norm of the gradient in the loss landscape.

Using the information content technique, we can get two types of bounds on the average of the norm of the gradient.
1. The bounds as achieved in the maximum Information Content regime: Gives us a lower and upper bound on the average norm of the gradient in case high Information Content is achieved.
2. The bounds as achieved in the sensitivity regime: Gives us an upper bound on the average norm of the gradient corresponding to the sensitivity IC achieved.

Thus, we get 3 bounds. The upper and lower bounds for the maximum IC and the upper bound for the sensitivity IC.

The `Trainer` class provides a method to calculate these gradient norms.

```python exec="on" source="material-block" html="1"
import torch
from torch.optim.adam import Adam

from qadence.constructors import ObservableConfig
from qadence.ml_tools.config import AnsatzConfig, FeatureMapConfig, TrainConfig
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.models import QNN
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.trainer import Trainer
from qadence.operations.primitive import Z

fm_config = FeatureMapConfig(num_features=1)
ansatz_config = AnsatzConfig(depth=4)
obs_config = ObservableConfig(detuning=Z)

qnn = QNN.from_configs(
    register=4,
    obs_config=obs_config,
    fm_config=fm_config,
    ansatz_config=ansatz_config,
)

optimizer = Adam(qnn.parameters(), lr=0.001)

batch_size = 25
x = torch.linspace(0, 1, 32).reshape(-1, 1)
y = torch.sin(x)
train_loader = to_dataloader(x, y, batch_size=batch_size, infinite=True)

train_config = TrainConfig(max_iter=100)

trainer = Trainer(
    model=qnn,
    optimizer=optimizer,
    config=train_config,
    loss_fn="mse",
    train_dataloader=train_loader,
    optimize_step=optimize_step,
)

# Perform exploratory landscape analysis with Information Content
ic_sensitivity_threshold = 1e-4
epsilons = torch.logspace(-2, 2, 10)

max_ic_lower_bound, max_ic_upper_bound, sensitivity_ic_upper_bound = (
    trainer.get_ic_grad_bounds(
        eta=ic_sensitivity_threshold,
        epsilons=epsilons,
    )
)

print(
    f"Using maximum IC, the gradients are bound between {max_ic_lower_bound:.3f} and {max_ic_upper_bound:.3f}\n"
)
print(
    f"Using sensitivity IC, the gradients are bounded above by {sensitivity_ic_upper_bound:.3f}"
)

# Resume training as usual...

trainer.fit(train_loader)
```

The `get_ic_grad_bounds` function returns a tuple containing a tuple containing the lower bound as achieved in maximum IC case, upper bound as achieved in maximum IC case, and the upper bound for the sensitivity IC case.

The sensitivity IC bound is guaranteed to appear, while the usually much tighter bounds that we get via the maximum IC case is only meaningful in the case of the maximum achieved information content $H(\epsilon)_{max} \geq log_6(2)$.



### 6.5. Custom `train` loop

If you need custom training functionality that goes beyond what is available in
`qadence.ml_tools.Trainer` you can write your own
training loop based on the building blocks that are available in Qadence.

A simplified version of Qadence's train loop is defined below. Feel free to copy it and modify at
will.

For logging we can use the `get_writer` from the `Writer Registry`. This will set up the default writer based on the experiment tracking tool.
All writers from the `Writer Registry` offer `open`, `close`, `print_metrics`, `write_metrics`, `plot_metrics`, etc methods.


```python
from typing import Callable, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qadence.ml_tools.config import TrainConfig
from qadence.ml_tools.data import DictDataLoader, data_to_device
from qadence.ml_tools.optimize_step import optimize_step
from qadence.ml_tools.callbacks import get_writer
from qadence.ml_tools.callbacks.saveload import load_checkpoint, write_checkpoint


def train(
    model: Module,
    data: DataLoader,
    optimizer: Optimizer,
    config: TrainConfig,
    loss_fn: Callable,
    device: str = "cpu",
    optimize_step: Callable = optimize_step,
    write_tensorboard: Callable = write_tensorboard,
) -> tuple[Module, Optimizer]:

    # Move model to device before optimizer is loaded
    model = model.to(device)

    # load available checkpoint
    init_iter = 0
    if config.log_folder:
        model, optimizer, init_iter = load_checkpoint(config.log_folder, model, optimizer)

    # Initialize writer based on the tracking tool specified in the configuration
    writer = get_writer(config.tracking_tool)  # Uses ExperimentTrackingTool to select writer
    writer.open(config, iteration=init_iter)

    dl_iter = iter(dataloader)

    # outer epoch loop
    for iteration in range(init_iter, init_iter + config.max_iter):
        data = data_to_device(next(dl_iter), device)
        loss, metrics = optimize_step(model, optimizer, loss_fn, data)

        if iteration % config.print_every == 0 and config.verbose:
            writer.print_metrics(OptimizeResult(iteration, model, optimizer, loss, metrics))

        if iteration % config.write_every == 0:
            writer.write(iteration, metrics)

        if config.log_folder:
            if iteration % config.checkpoint_every == 0:
                write_checkpoint(config.log_folder, model, optimizer, iteration)

    # Final writing and checkpointing
    if config.log_folder:
        write_checkpoint(config.log_folder, model, optimizer, iteration)
    writer.write(iteration,metrics)
    writer.close()

    return model, optimizer
```

### 6.6. Gradient-free optimization using `Trainer`

We can achieve gradient free optimization with `Trainer.set_use_grad(False)` or `trainer.disable_grad_opt(ng_optimizer)`. An example solving a QUBO using gradient free optimization based on `Nevergrad` optimizers and `Trainer` is shown in the [analog QUBO Tutorial](../../digital_analog_qc/analog-qubo.md).
