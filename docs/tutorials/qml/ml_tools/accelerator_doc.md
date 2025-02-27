# Accelerator for Distributed Training

## Overview

The `Accelerator` class is designed to simplify distributed training with PyTorch's API. It allows for efficient training across multiple GPUs or processes while handling device placement, data distribution, and model synchronization. It uses `DistDataParallel` and `DistributedSampler` in the background to correctly distribute the model and training data across processes and devices.

This tutorial will guide you through setting up and using `Accelerator` for distributed training.

## Accelerator

The `Accelerator` class manages the training environment and process distribution. Here’s how you initialize it:

```python
from qadence.ml_tools.train_utils import Accelerator
import torch

accelerator = Accelerator(
    nprocs=4,               # Number of processes (e.g., GPUs). Enables multiprocessing.
    compute_setup="auto",   # Automatically selects available compute devices
    log_setup="cpu",        # Logs on CPU to avoid memory overhead
    dtype=torch.float32,    # Data type for numerical precision
    backend="nccl"          # Backend for communication
)
```

#### Using Accelerator with Trainer

`Accelerator` is already integrated into the `Trainer` class from `qadence.ml_tools`, and `Trainer` can automatically distribute the training process based on the configurations provided in `TrainConfig`.

```python
from qadence.ml_tools.trainer import Trainer
from qadence.ml_tools import TrainConfig

config = TrainConfig(nprocs=4)

trainer = Trainer(model, optimizer, config)
model, optimizer = trainer.fit(dataloader)
```


### Accelerator features

The `Accelerator` also provides a `distribute()` function wrapper that simplifies running distributed training across multiple processes. This method can be used to prepare or wrap a function that needs to be distributed.

-  `distribute()`

    This method allows you to wrap your training function so it runs across multiple processes, handling rank management and process spawning automatically.

    **Example Usage**:
    ```python
    distributed_fun = accelerator.distribute(fun)
    distributed_fun(*args, **kwargs)
    ```

    The `distribute()` function ensures that each process runs on a designated device and synchronizes properly, making it easier to scale training with minimal code modifications.

The `Accelerator` further offers these key methods: `prepare`, `prepare_batch`, and `all_reduce_dict`.


- `prepare()`

    This method ensures that models, optimizers, and dataloaders are properly placed on the correct devices for distributed training. It wraps models into `DistributedDataParallel` and synchronizes parameters across processes.

    ```python
    model, optimizer, dataloader = accelerator.prepare(model,
                                                        optimizer,
                                                        dataloader)
    ```

- `prepare_batch()`
    Moves data batches to the correct device and formats them properly for distributed training.

    ```python
    batch_data, batch_targets = accelerator.prepare(batch)
    ```

- `all_reduce_dict()`
    Aggregates and synchronizes metrics across all processes during training. Note: This will cause a synchronization overhead and slow down the training processes.

    ```python
    metrics = {"loss": torch.tensor(1.0)}
    reduced_metrics = accelerator.all_reduce_dict(metrics)
    print(reduced_metrics)
    ```

## Example

To launch distributed training across multiple GPUs/CPUs, use the following approach:
Each batch should be moved to the correct device. The `prepare_batch()` method simplifies this process.

### Example Code (train_script.py):
```python exec="on" source="material-block" html="1"
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools.train_utils import Accelerator


def train_epoch(epochs, model, dataloader, optimizer, accelerator):

    # Prepare model, optimizer, and dataloader for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # accelerator.rank will provide you the rank of the process.
    if accelerator.rank == 0:
        print("Prepared model of type: ", type(model))
        print("Prepared optimizer of type: ", type(optimizer))
        print("Prepared dataloader of type: ", type(dataloader))

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:

            # Move batch to the correct device
            batch = accelerator.prepare_batch(batch)

            batch_data, batch_targets = batch
            optimizer.zero_grad()
            output = model(batch_data)
            loss = torch.nn.functional.mse_loss(output, batch_targets)
            loss.backward()
            optimizer.step()
        print("Rank: ", accelerator.rank, " | Epoch: ", epoch, " | Loss: ", loss.item())

if __name__ == "__main__":

    model = nn.Sequential(
        nn.Linear(10, 100),  # Input Layer
        nn.ReLU(),  # Activation Function
        nn.Linear(100, 1)  # Output Layer
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # A random dataset with 10 features and a target to predict.
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    dataloader = DataLoader(dataset, batch_size=32)

    accelerator = Accelerator(
        nprocs=4,               # Number of processes (e.g., GPUs). Enables multiprocessing.
        compute_setup="cpu",    # or choose GPU
        backend="gloo"          # choose `nccl` for GPU
    )

    distributed_train_epoch = accelerator.distribute(train_epoch)
    distributed_train_epoch(n_epochs, model, dataloader, optimizer, accelerator)
```

### Running Distributed Training

The above example can be directly run on the terminal as following:

```bash
python train_script.py
```

- **SLURM**:

    To launch distributed training across multiple GPUs

    Inside an interactive `srun` session, you can directly use:
    ```bash
    python train_script.py
    ```

    Or submit the following sbatch script:
    ```bash
    #!/bin/bash
    #SBATCH --job-name=MG_slurm
    #SBATCH --nodes=1
    #SBATCH --ntasks=1              # tasks = number of nodes
    #SBATCH --gpus-per-task=4       # same as nprocs
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=56G

    srun python3 train_script.py
    ```

- **Torchrun**:

    To run distributed training with `torchrun`
    ```bash
    #!/bin/bash
    #SBATCH --job-name=MG_torchrun
    #SBATCH --nodes=1
    #SBATCH --ntasks=1              # tasks = number of nodes
    #SBATCH --gpus-per-task=2       # same as nprocs
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=56G

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')

    srun torchrun --nnodes 1 --nproc_per_node 2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29522 train_script.py
    ```

## Conclusion

The `Accelerator` class simplifies distributed training by handling process management, device setup, and data distribution. By integrating it into your PyTorch training workflow, you can efficiently scale training across multiple devices with minimal code modifications.
