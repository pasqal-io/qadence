# Training on GPU with `Trainer`

This guide explains how to train models on **GPU** using `Trainer` from `qadence.ml_tools`, covering **single-GPU**, **multi-GPU (single node)**, and **multi-node multi-GPU** setups.

### Understanding Arguments
- *nprocs*: Number of processes to run. To enable multi-processing and launch separate processes, set nprocs > 1.
- *compute_setup*: The computational setup used for training. Options include `cpu`, `gpu`, and `auto`.

For more details on the advanced training options, please refer to [TrainConfig Documentation](./data_and_config.md)

## **Configuring `TrainConfig` for GPU Training**
By adjusting `TrainConfig`, you can switch between single and multi-GPU training setups. Below are the key settings for each configuration:

### **Single-GPU Training Configuration:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Optimized backend for GPU training.
- **`nprocs=1`**: Uses one GPU.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=1,
)
```

### **Multi-GPU (Single Node) Training Configuration:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Multi-GPU optimized backend.
- **`nprocs=2`**: Utilizes 2 GPUs on a single node.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=2,
)
```

### **Multi-Node Multi-GPU Training Configuration:**
- **`compute_setup`**: Selected training setup. (`gpu` or `auto`)
- **`backend="nccl"`**: Required for multi-node setups.
- **`nprocs=4`**: Uses 4 GPUs across nodes.
```python
train_config = TrainConfig(
    compute_setup="auto",
    backend="nccl",
    nprocs=4,
)
```
---

## Examples

The following sections provide Python scripts and training approach scripts for each setup.


> Some organizations use [SLURM](https://slurm.schedmd.com) to manage resources. Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. If you are using slurm, you can use the `Trainer` by submitting a batch script using sbatch. Further below, we also provide the sbatch scripts for each setup.

> You can also use `torchrun` to run the training process - which provides a superset of the functionality as `torch.distributed.launch `. Here you need to specify the [torchrun arguments](https://pytorch.org/docs/stable/elastic/run.html) arguments to set up the distributed training setup. We also include the `torchrun` sbatch scripts for each setup below.

### Example Training Script (`train.py`):

We are going to use the following training script for the examples below.
**Python Script:**
```python
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from qadence.ml_tools import TrainConfig, Trainer
from qadence.ml_tools.optimize_step import optimize_step
Trainer.set_use_grad(True)

# __main__ is recommended.
if __name__ == "__main__":
    # simple dataset for y = 2πx
    x = torch.linspace(0, 1, 100).reshape(-1, 1)
    y = torch.sin(2 * torch.pi * x)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
    # Simple model with no hidden layer and ReLU activation to fit the data for y = 2πx
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    # SGD optimizer with 0.01 learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # TrainConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int,
                        default=1, help="Number of processes (GPUs) to use.")
    parser.add_argument("--compute_setup", type=str,
                        default="auto", choices=["cpu", "gpu", "auto"], help="Computational Setup.")
    parser.add_argument("--backend", type=str,
                        default="nccl", choices=["nccl", "gloo", "mpi"], help="Distributed backend.")
    args = parser.parse_args()
    train_config = TrainConfig(
                                backend=args.backend,
                                nprocs=args.nprocs,
                                compute_setup=args.compute_setup,
                                print_every=5,
                                max_iter=50
                            )

    trainer = Trainer(model, optimizer, train_config, loss_fn="mse", optimize_step=optimize_step)
    trainer.fit(dataloader)
```

---
### 1. Single-GPU:

Simple and suitable for single-card setups.
- *Assuming that you have 1 node with 1 GPU.*

You can train by calling this on the head node.
```bash
python3 train.py --backend nccl --nprocs 1
```

#### SLURM
Slurm can be used to train to train the model.
```bash
#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

srun python3 train.py --backend nccl --nprocs 1
```

#### TORCHRUN
Torchrun takes care of setting the `nprocs` based on the cluster setup. We only need to specify to use the `compute_setup`, which can be either `auto` or `gpu`.
- `nnodes` for torchrun should be the number of nodes
- `nproc_per_node` should be equal to the number of GPUs per node.

> Note: We use the first node of the allocated resources on the cluster as the head node. However, any other node can also be chosen.
```bash
#!/bin/bash
#SBATCH --job-name=single_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
export LOGLEVEL=INFO

srun torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29501 \
train.py --compute_setup auto
```

### 2. Multi-GPU (Single Node):

For high performance using multiple GPUs in one node.
- *Assuming that you have 1 node with 2 GPU. These numbers can be changed depending on user needs.*

You can train by simply calling this on the head node.
```bash
python3 train.py --backend nccl --nprocs 2
```

#### SLURM
Slurm can be used to train the model but also to dispatch the workload on multiple GPUs or CPUs.
- Here, we should have one task per gpu. i.e. `ntasks` is equal to the number of nodes
- `nprocs` should be equal to the total number of gpus. which is this case is 2.

```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

srun python3 train.py --backend nccl --nprocs 2
```

#### TORCHRUN
Torchrun takes care of setting the `nprocs` based on the cluster setup. We only need to specify to use the `compute_setup`, which can be either `auto` or `gpu`.
- `nnodes` for torchrun should be the number of nodes
- `nproc_per_node` should be equal to the number of GPUs per node.

> Note: We use the first node of the allocated resources on the cluster as the head node. However, any other node can also be chosen.
```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
export LOGLEVEL=INFO

srun torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29501 \
train.py --compute_setup auto
```

### 3. Multi-Node Multi-GPU:

For high performance using multiple GPUs in multiple nodes.
- *Assuming that you have two nodes with two GPU each. These numbers can be customised on user needs.*

For multi-node, it is suggested to submit a sbatch script.

#### SLURM
- We should have one task per gpu. i.e. `ntasks` is equal to the number of nodes.
- `nprocs` should be equal to the total number of gpus. which is this case is 4.

```bash
#!/bin/bash
#SBATCH --job-name=multi_node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

srun python3 train.py --backend nccl --nprocs 4
```


#### TORCHRUN
Torchrun takes care of setting the `nprocs` based on the cluster setup. We only need to specify to use the `compute_setup`, which can be either `auto` or `gpu`.
- `nnodes` for torchrun should be the number of nodes
- `nproc_per_node` should be equal to the number of GPUs per node.

> Note: We use the first node of the allocated resources on the cluster as the head node. However, any other node can also be chosen.
```bash
#!/bin/bash
#SBATCH --job-name=multi_node
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
export LOGLEVEL=INFO

srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29501 \
train.py --compute_setup auto
```
