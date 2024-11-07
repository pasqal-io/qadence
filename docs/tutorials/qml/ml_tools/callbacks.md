
# Callbacks for Trainer

Qadence ml_tools provides a powerful callback system for customizing various stages of the training process. With callbacks, you can monitor, log, save, and alter your training workflow efficiently. A `CallbackManager` is used with [`Trainer`][qadence.ml_tools.Trainer] to execute the training process with defined callbacks. The default callbacks already provided in the [`Trainer`][qadence.ml_tools.Trainer].

### Default Callbacks
Below is a list of the default callbacks already implemented in the `CallbackManager` used with [`Trainer`][qadence.ml_tools.Trainer]:

- **`on_train_start`**: `PlotMetrics`, `SaveCheckpoint`, `WriteMetrics`
- **`on_train_epoch_end`**: `SaveCheckpoint`, `PrintMetrics`, `PlotMetrics`, `WriteMetrics`
- **`on_val_epoch_end`**: `SaveBestCheckpoint`, `WriteMetrics`
- **`on_train_end`**: `LogHyperparameters`, `LogModelTracker`, `WriteMetrics`, `SaveCheckpoint`

This guide covers how to define and use callbacks in `TrainConfig`, integrate them with the `Trainer` class, and create custom callbacks using hooks.


## 1. Built-in Callbacks

Qadence ml_tools offers several built-in callbacks for common tasks like saving checkpoints, logging metrics, and tracking models. Below is an overview of each.

### 1.1. `PrintMetrics`

Prints metrics at specified intervals.

```python
from qadence.ml_tools import TrainConfig
from qadence.ml_tools.callbacks import PrintMetrics

print_metrics_callback = PrintMetrics(on="on_val_batch_end", called_every=100)

config = TrainConfig(
    max_iter=10000,
    callbacks=[print_metrics_callback]
)
```

### 1.2. `WriteMetrics`

Writes metrics to a specified logging destination.

```python
from qadence.ml_tools.callbacks import WriteMetrics

write_metrics_callback = WriteMetrics(on="on_train_epoch_end", called_every=50)

config = TrainConfig(
    max_iter=5000,
    callbacks=[write_metrics_callback]
)
```

### 1.3. `PlotMetrics`

Plots metrics based on user-defined plotting functions.

```python
from qadence.ml_tools.callbacks import PlotMetrics

plot_metrics_callback = PlotMetrics(on="on_train_epoch_end", called_every=100)

config = TrainConfig(
    max_iter=5000,
    callbacks=[plot_metrics_callback]
)
```

### 1.4. `LogHyperparameters`

Logs hyperparameters to keep track of training settings.

```python
from qadence.ml_tools.callbacks import LogHyperparameters

log_hyper_callback = LogHyperparameters(on="on_train_start", called_every=1)

config = TrainConfig(
    max_iter=1000,
    callbacks=[log_hyper_callback]
)
```

### 1.5. `SaveCheckpoint`

Saves model checkpoints at specified intervals.

```python
from qadence.ml_tools.callbacks import SaveCheckpoint

save_checkpoint_callback = SaveCheckpoint(on="on_train_epoch_end", called_every=100)

config = TrainConfig(
    max_iter=10000,
    callbacks=[save_checkpoint_callback]
)
```

### 1.6. `SaveBestCheckpoint`

Saves the best model checkpoint based on a validation criterion.

```python
from qadence.ml_tools.callbacks import SaveBestCheckpoint

save_best_checkpoint_callback = SaveBestCheckpoint(on="on_val_epoch_end", called_every=10)

config = TrainConfig(
    max_iter=10000,
    callbacks=[save_best_checkpoint_callback]
)
```

### 1.7. `LoadCheckpoint`

Loads a saved model checkpoint at the start of training.

```python
from qadence.ml_tools.callbacks import LoadCheckpoint

load_checkpoint_callback = LoadCheckpoint(on="on_train_start")

config = TrainConfig(
    max_iter=10000,
    callbacks=[load_checkpoint_callback]
)
```

### 1.8. `LogModelTracker`

Logs the model structure and parameters.

```python
from qadence.ml_tools.callbacks import LogModelTracker

log_model_callback = LogModelTracker(on="on_train_end")

config = TrainConfig(
    max_iter=1000,
    callbacks=[log_model_callback]
)
```


## 2. Custom Callbacks

The base `Callback` class in Qadence allows defining custom behavior that can be triggered at specified events (e.g., start of training, end of epoch). You can set parameters such as when the callback runs (`on`), frequency of execution (`called_every`), and optionally define a `callback_condition`.

### defining Callbacks

There are two main ways to define a callback:
1. **Directly providing a function** in the `Callback` instance.
2. **Subclassing** the `Callback` class and implementing custom logic.

#### Example 1: Providing a Callback Function Directly

```python
from qadence.ml_tools.callbacks import Callback

# Define a custom callback function
def custom_callback_function(trainer, config, writer):
    print("Executing custom callback.")

# Create the callback instance
custom_callback = Callback(
    on="on_train_end",
    called_every=5,
    callback=custom_callback_function
)
```

#### Example 2: Subclassing the Callback

```python
from qadence.ml_tools.callbacks import Callback

class CustomCallback(Callback):
    def run_callback(self, trainer, config, writer):
        print("Custom behavior in run_callback method.")

# Create the subclassed callback instance
custom_callback = CustomCallback(on="on_train_end", called_every=10)
```


## 3. Adding Callbacks to `TrainConfig`

To use callbacks in `TrainConfig`, add them to the `callbacks` list when configuring the training process.

```python
from qadence.ml_tools import TrainConfig
from qadence.ml_tools.callbacks import SaveCheckpoint, PrintMetrics

config = TrainConfig(
    max_iter=10000,
    callbacks=[
        SaveCheckpoint(on="on_val_epoch_end", called_every=50),
        PrintMetrics(on="on_train_epoch_end", called_every=100),
    ]
)
```

## 4. Using Callbacks with `Trainer`

The `Trainer` class in `qadence.ml_tools` provides built-in support for executing callbacks at various stages in the training process, managed through a callback manager. By default, several callbacks are added to specific hooks to automate common tasks, such as check-pointing, metric logging, and model tracking.

### Default Callbacks
Below is a list of the default callbacks and their assigned hooks:

- **`on_train_start`**: `PlotMetrics`, `SaveCheckpoint`, `WriteMetrics`
- **`on_train_epoch_end`**: `SaveCheckpoint`, `PrintMetrics`, `PlotMetrics`, `WriteMetrics`
- **`on_val_epoch_end`**: `SaveBestCheckpoint`, `WriteMetrics`
- **`on_train_end`**: `LogHyperparameters`, `LogModelTracker`, `WriteMetrics`, `SaveCheckpoint`

These defaults handle common needs, but you can also add custom callbacks to any hook.

### Example: Adding a Custom Callback

To create a custom `Trainer` that includes a `PrintMetrics` callback executed specifically at the end of each epoch, follow the steps below.


```python
from qadence.ml_tools.trainer import Trainer
from qadence.ml_tools.callback import PrintMetrics

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_metrics_callback = PrintMetrics(on="on_train_epoch_end", called_every = 10)

    def on_train_epoch_end(self, train_epoch_loss_metrics):
        self.print_metrics_callback.run_callback(self)
```
