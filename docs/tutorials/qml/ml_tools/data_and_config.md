## 1. Dataloaders

When using Qadence, you can supply classical data to a quantum machine learning
algorithm by using a standard PyTorch `DataLoader` instance. Qadence also provides
the `DictDataLoader` convenience class which allows
to build dictionaries of `DataLoader`s instances and easily iterate over them.

```python exec="on" source="material-block" result="json"
import torch
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import DictDataLoader, to_dataloader


def dataloader(data_size: int = 25, batch_size: int = 5, infinite: bool = False) -> DataLoader:
    x = torch.linspace(0, 1, data_size).reshape(-1, 1)
    y = torch.sin(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=infinite)


def dictdataloader(data_size: int = 25, batch_size: int = 5) -> DictDataLoader:
    dls = {}
    for k in ["y1", "y2"]:
        x = torch.rand(data_size, 1)
        y = torch.sin(x)
        dls[k] = to_dataloader(x, y, batch_size=batch_size, infinite=True)
    return DictDataLoader(dls)


# iterate over standard DataLoader
for (x,y) in dataloader(data_size=6, batch_size=2):
    print(f"Standard {x = }")

# construct an infinite dataset which will keep sampling indefinitely
n_epochs = 5
dl = iter(dataloader(data_size=6, batch_size=2, infinite=True))
for _ in range(n_epochs):
    (x, y) = next(dl)
    print(f"Infinite {x = }")

# iterate over DictDataLoader
ddl = dictdataloader()
data = next(iter(ddl))
print(f"{data = }")
```

Note:
    In case of `infinite`=True, the dataloader iterator will provide a random sample from the dataset.

## 2. Training Configuration

The [`TrainConfig`][qadence.ml_tools.config.TrainConfig] class provides a comprehensive configuration setup for training quantam machine learning models in Qadence. This configuration includes settings for batch size, logging, check-pointing, validation, and additional custom callbacks that control the training process's granularity and flexibility.

The [`TrainConfig`][qadence.ml_tools.config.TrainConfig] tells [`Trainer`][qadence.ml_tools.Trainer]  what batch_size should be used, how many epochs to train, in which intervals to print/log metrics and how often to store intermediate checkpoints.
It is also possible to provide custom callback functions by instantiating a [`Callback`][qadence.ml_tools.callbacks.Callback]
with a function `callback`.

For example of how to use the TrainConfig with `Trainer`, please see [Examples in Trainer](./trainer.md)


### 2.1 Explanation of `TrainConfig` Attributes

| Attribute                | Type                     | Default                  | Description |
|--------------------------|--------------------------|--------------------------|-------------|
| `max_iter`               | `int`                    | `10000`                  | Total number of training epochs. |
| `batch_size`             | `int`                    | `1`                      | Batch size for training. |
| `print_every`            | `int`                    | `0`                   | Frequency of console output. Set to `0` to disable. |
| `write_every`            | `int`                    | `0`                     | Frequency of logging metrics. Set to `0` to disable. |
| `plot_every`             | `int`                    | `0`                   | Frequency of plotting metrics. Set to `0` to disable. |
| `checkpoint_every`       | `int`                    | `0`                   | Frequency of saving checkpoints. Set to `0` to disable. |
| `val_every`              | `int`                    | `0`                      | Frequency of validation checks. Set to `0` to disable. |
| `val_epsilon`            | `float`                  | `1e-5`                   | Threshold for validation improvement. |
| `validation_criterion`   | `Callable`               | `None`                   | Function for validating metric improvement. |
| `trainstop_criterion`    | `Callable`               | `None`                   | Function to stop training early. |
| `callbacks`              | `list[Callback]`         | `[]`                     | List of custom callbacks. |
| `root_folder`            | `Path`                   | `"./qml_logs"`           | Root directory for saving logs and checkpoints. |
| `log_folder`             | `Path`                   | `"./qml_logs"`           | Logging directory for saving logs and checkpoints. |
| `log_model`              | `bool`                   | `False`                  | Enables model logging. |
| `verbose`                | `bool`                   | `True`                   | Enables detailed logging. |
| `tracking_tool`          | `ExperimentTrackingTool` | `TENSORBOARD`            | Tool for tracking training metrics. |
| `plotting_functions`     | `tuple`                  | `()`                     | Functions for plotting metrics. |
| `hyperparams`            | `dict`                   | `{}`                     | Dictionary of hyperparameters |
| `nprocs`                 | `int`                    | `1`               | Number of processes to use when spawning subprocesses; for multi-GPU setups, set this to the total number of GPUs. |
| `compute_setup`          | `str`                    | `"cpu"`                  | Specifies the compute device: `"auto"`, `"gpu"`, or `"cpu"`.|
| `backend`                | `str`                    | `"gloo"`                 | Backend for distributed training communication (e.g., `"gloo"`, `"nccl"`, or `"mpi"`). |
| `log_setup`              | `str`                    | `"cpu"`                  | Device setup for logging; use `"cpu"` to avoid GPU conflicts|
| `dtype`                  | `dtype` or `None`        | `None`                   | Data type for computations (e.g., `torch.float32`) |
| `all_reduce_metrics`     | `bool`                   | `False`                  | If `True`, aggregates metrics (e.g., loss) across processes |



```python exec="on" source="material-block"
from qadence.ml_tools import OptimizeResult, TrainConfig
from qadence.ml_tools.callbacks import Callback

batch_size = 5
n_epochs = 100

print_parameters = lambda opt_res: print(opt_res.model.parameters())
condition_print = lambda opt_res: opt_res.loss < 1.0e-03
modify_extra_opt_res = {"n_epochs": n_epochs}
custom_callback = Callback(on="train_end", callback = print_parameters, callback_condition=condition_print, modify_optimize_result=modify_extra_opt_res, called_every=10,)

config = TrainConfig(
    root_folder="some_path/",
    max_iter=n_epochs,
    checkpoint_every=100,
    write_every=100,
    batch_size=batch_size,
    callbacks = [custom_callback]
)
```


### 2.2 Key Configuration Options in `TrainConfig`

#### Iterations and Batch Size

- `max_iter` (**int**): Specifies the total number of training iterations (epochs). For an `InfiniteTensorDataset`, each epoch contains one batch; for a `TensorDataset`, it contains `len(dataloader)` batches.
- `batch_size` (**int**): Defines the number of samples processed in each training iteration.

Example:
```python
config = TrainConfig(max_iter=2000, batch_size=32)
```

#### Training Parameters

- `print_every` (**int**): Controls how often loss and metrics are printed to the console.
- `write_every` (**int**): Determines how frequently metrics are written to the tracking tool, such as TensorBoard or MLflow.
- `checkpoint_every` (**int**): Sets the frequency for saving model checkpoints.

Note: Set 0 to diable.

Example:
```python
config = TrainConfig(print_every=100, write_every=50, checkpoint_every=50)
```

The user can provide either the `root_folder` or the `log_folder` for saving checkpoints and logging. When neither are provided, the default `root_folder` "./qml_logs" is used.

- `root_folder` (**Path**): The root directory for saving checkpoints and logs. All training logs will be saved inside a subfolder in this root directory. (The path to these subfolders can be accessed using config._subfolders, and the current logging folder is config.log_folder)
- `create_subfolder_per_run` (**bool**): Creates a unique subfolder for each training run within the specified folder.
- `tracking_tool` (**ExperimentTrackingTool**): Specifies the tracking tool to log metrics, e.g., TensorBoard or MLflow.
- `log_model` (**bool**): Enables logging of a serialized version of the model, which is useful for model versioning. Thi happens at the end of training.

Note
    - The user can also provide `log_folder` argument - which will only be used when `create_subfolder_per_run` = False.
    -  `log_folder` (**Path**): The log folder used for saving checkpoints and logs.

Example:
```python
config = TrainConfig(root_folder="path/to/checkpoints", tracking_tool=ExperimentTrackingTool.MLFLOW, checkpoint_best_only=True)
```

#### Validation Parameters

- `checkpoint_best_only` (**bool**): If set to `True`, saves checkpoints only when there is an improvement in the validation metric.
- `val_every` (**int**): Frequency of validation checks. Setting this to `0` disables validation.
- `val_epsilon` (**float**): A small threshold used to compare the current validation loss with previous best losses.
- `validation_criterion` (**Callable**): A custom function to assess if the validation metric meets a specified condition.

Example:
```python
config = TrainConfig(val_every=200, checkpoint_best_only = True, validation_criterion=lambda current, best: current < best - 0.001)
```

If it is desired to only the save the "best" checkpoint, the following must be ensured:

    (a) `checkpoint_best_only = True` is used while creating the configuration through `TrainConfig`,
    (b) `val_every` is set to a valid integer value (for example, `val_every = 10`) which controls the no. of iterations after which the validation data should be used to evaluate the model during training, which can also be set through `TrainConfig`,
    (c) a validation criterion is provided through the `validation_criterion`, set through `TrainConfig` to quantify the definition of "best", and
    (d) the validation dataloader passed to `Trainer` is of type `DataLoader`. In this case, it is expected that a validation dataloader is also provided along with the train dataloader since the validation data will be used to decide the "best" checkpoint.

The criterion used to decide the "best" checkpoint can be customized by `validation_criterion`, which should be a function that can take val_loss, best_loss, and val_epsilon arguments and return a boolean value (True or False) indicating whether some validation metric is satisfied or not. An example of a simple `validation_criterion` is:
```python
def validation_criterion(val_loss: float, best_val_loss: float, val_epsilon: float) -> bool:
    return val_loss < (best_val_loss - val_epsilon)
```

#### Custom Callbacks

`TrainConfig` supports custom callbacks that can be triggered at specific stages of training. The `callbacks` attribute accepts a list of callback instances, which allow for custom behaviors like early stopping or additional logging.
See [Callbacks](./callbacks.md) for more details.

- `callbacks` (**list[Callback]**): List of custom callbacks to execute during training.

Example:
```python
from qadence.ml_tools.callbacks import Callback

def callback_fn(trainer, config, writer):
    if trainer.opt_res.loss < 0.001:
        print("Custom Callback: Loss threshold reached!")

custom_callback = Callback(on = "train_epoch_end", called_every = 10, callback_function = callback_fn )

config = TrainConfig(callbacks=[custom_callback])
```

#### Hyperparameters and Plotting

- `hyperparams` (**dict**): A dictionary of hyperparameters (e.g., learning rate, regularization) to be tracked by the tracking tool.
- `plot_every` (**int**): Determines how frequently plots are saved to the tracking tool, such as TensorBoard or MLflow.
- `plotting_functions` (**tuple[LoggablePlotFunction, ...]**): Functions for in-training plotting of metrics or model state.

Note: Please ensure that plotting_functions are provided when plot_every > 0

Example:
```python
config = TrainConfig(
    plot_every=10,
    hyperparams={"learning_rate": 0.001, "batch_size": 32},
    plotting_functions=(plot_loss_function,)
)
```

#### Advanced Distributed Training

- `nprocs` (**int**): Specifies the number of processes to be used. For multi-GPU training, this should match the total number of GPUs available. When nprocs is greater than 1, `Trainer` spawns additional subprocesses for training. This is useful for parallel or distributed training setups.

- `compute_setup` (**str**): Determines the compute device configuration: 1.`"auto"` (automatically selects GPU if available), 2. `"gpu"` - (forces GPU usage and errors if no GPU is detected), and 3. `"cpu"` (Forces the use of the CPU).

- `backend` (**str**): Specifies the communication backend for distributed training. Common options are `"gloo"` (default), `"nccl"` (optimized for GPUs), or `"mpi"`, depending on your setup. It should be one of the backends supported by `torch.distributed`. For further details, please look at [torch backends](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend)


> Notes:
> - *Logging Specific Callbacks*: Logging is available only through the main process, i.e. process 0.  Model logging, plotting, logging metrics will only be performed for a single process, even if multiple processes are run.
> - *Training with specific callbacks*: Callbacks specific to training, e.g., `EarlyStopping`, `LRSchedulerStepDecay`, etc will be called from each process.
> - `PrintMetrics` (set through the `print_every` argument in `TrainCongig`) is available from all processes.


Example: For CPU MultiProcessing
```python
config = TrainConfig(
    compute_setup="cpu",
    nprocs=5,
    backend="gloo"
)
```

Example: For GPU multiprocessing training
```python
config = TrainConfig(
    compute_setup="gpu",
    nprocs=2, # World-size/Total number of GPUs
    backend="nccl"
)
```

#### Precision Options

- `dtype` (**dtype** or **None**): Sets the numerical precision (data type) for computations. For instance, you can use `torch.float32` or `torch.float16` depending on your performance and precision needs. Both model parameters, and dataset will be of the provided precision.
    - If not specified or None, the default torch precision (usually torch.float32) is used.
    - If provided dtype is complex dtype, appropriate precision for the data and model parameters will be used as follows:

    | Data Type (`dtype`)   | Data Precision | Model Precision | Model Parameters Precision  (*Real Part*  & *Imaginary Part* )|
    |---------------------|---------------|----------------|-------------------------------------|
    | `torch.float16`     | 16-bit        | 16-bit         | N/A            | N/A                |
    | `torch.float32`     | 32-bit        | 32-bit         | N/A            | N/A                |
    | `torch.float64`     | 64-bit        | 64-bit         | N/A            | N/A                |
    | `torch.complex32`   | 16-bit        | 32-bit         | 16-bit         | 16-bit             |
    | `torch.complex64`   | 32-bit        | 64-bit         | 32-bit         | 32-bit             |
    | `torch.complex128`  | 64-bit        | 128-bit        | 64-bit         | 64-bit             |

    **Complex Dtypes**: Complex data types are useful for Quantum Neural Networks - such as `QNN` provided by qadence. The industry standard is to use `torch.complex128`, however, the user can also specify a lower precision (`torch.complex64` or  `torch.complex32`) for faster training.



Furthermore, the user can also utilize the following options:

- `log_setup` (**str**): Configures the device used for logging. Using `"cpu"` ensures logging runs on the CPU (which may avoid conflicts with GPU operations), while `"auto"` aligns logging with the compute device.

- `all_reduce_metrics` (**bool**): When enabled, aggregates metrics (such as loss or accuracy) across all training processes to provide a unified summary, though it may introduce additional synchronization overhead.

## 3. Experiment tracking with mlflow

Qadence allows to track runs and log hyperparameters, models and plots with [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) and [mlflow](https://mlflow.org/). In the following, we demonstrate the integration with mlflow.

### mlflow configuration
We have control over our tracking configuration by setting environment variables. First, let's look at the tracking URI. For the purpose of this demo we will be working with a local database, in a similar fashion as described [here](https://mlflow.org/docs/latest/tracking/tutorials/local-database.html),
```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

Qadence can also read the following two environment variables to define the mlflow experiment name and run name
```bash
export MLFLOW_EXPERIMENT=test_experiment
export MLFLOW_RUN_NAME=run_0
```

If no tracking URI is provided, mlflow stores run information and artifacts in the local `./mlflow` directory and if no names are defined, the experiment and run will be named with random UUIDs.
